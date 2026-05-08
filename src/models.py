# src/models.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

# ── Tabular (XGBoost) ─────────────────────────────────────────────────────────
def fit_dual_tabular_model(X_train, y_drop, y_fail,
                           name="xgboost", random_state=42):
    clf = MultiOutputClassifier(
        XGBClassifier(n_estimators=300, max_depth=6,
                      learning_rate=0.05, use_label_encoder=False,
                      eval_metric="logloss", random_state=random_state,
                      verbosity=0)
    )
    clf.fit(X_train, np.column_stack([y_drop, y_fail]))
    return clf

def predict_dual_tabular_model(model, X_test):
    proba = model.predict_proba(X_test)
    p_drop = proba[0][:, 1]
    p_fail = proba[1][:, 1]
    return p_drop, p_fail


# ── Shared LSTM ───────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_dim=2,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


# ── Shared Transformer ────────────────────────────────────────────────────────
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_dim=2,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj  = nn.Linear(input_dim, hidden_dim)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        return self.head(x[:, -1, :])


# ── MultiTask Causal Net ──────────────────────────────────────────────────────
class MultiTaskCausalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64,
                 out_dim=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.drop_head = nn.Linear(hidden_dim, 1)
        self.fail_head = nn.Linear(hidden_dim, 1)
        self.causal_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        rep = h[-1]
        return self.drop_head(rep).squeeze(-1), self.fail_head(rep).squeeze(-1)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_torch_model(model, X_train, y_train,
                      epochs=15, lr=1e-3, batch_size=128,
                      causal_targets=None, causal_lambda=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)

    if yt.ndim == 2:
        pw0 = float((yt[:, 0] == 0).sum()) / max(1, float((yt[:, 0] == 1).sum()))
        pw1 = float((yt[:, 1] == 0).sum()) / max(1, float((yt[:, 1] == 1).sum()))
        bce_drop = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw0], dtype=torch.float32).to(device))
        bce_fail = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw1], dtype=torch.float32).to(device))
    else:
        pw = float((yt == 0).sum()) / max(1, float((yt == 1).sum()))
        bce_drop = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], dtype=torch.float32).to(device))
        bce_fail = bce_drop

    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if isinstance(out, tuple):
                loss = bce_drop(out[0], yb[:, 0]) + bce_fail(out[1], yb[:, 1])
            else:
                loss = bce_drop(out, yb)
            if causal_targets is not None and causal_lambda > 0:
                ct = torch.tensor(
                    causal_targets, dtype=torch.float32).to(device)
                if hasattr(model, "causal_proj"):
                    _, (h, _) = model.encoder(xb)
                    reg = ((model.causal_proj(h[-1]) -
                            ct[:xb.size(0)].unsqueeze(1).expand_as(
                                model.causal_proj(h[-1]))
                            ) ** 2).mean()
                    loss = loss + causal_lambda * reg
            loss.backward()
            opt.step()
    return model


def predict_torch_multitask(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        xt  = torch.tensor(X_test, dtype=torch.float32).to(device)
        out = model(xt)
    if isinstance(out, tuple):
        pd_ = torch.sigmoid(out[0]).cpu().numpy()
        pf_ = torch.sigmoid(out[1]).cpu().numpy()
        return pd_, pf_
    return torch.sigmoid(out).cpu().numpy()

# ── aliases ──────────────────────────────────────────────
fitdualtabularmodel     = fit_dual_tabular_model
predictdualtabularmodel = predict_dual_tabular_model
traintorchmodel         = train_torch_model
predicttorchmultitask   = predict_torch_multitask