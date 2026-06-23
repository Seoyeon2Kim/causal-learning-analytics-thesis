# Causal Learning Analytics for Early Dropout and Failure Risk Detection in Online Education

This repository contains the reproducible code, figures, and result tables for the journal paper and thesis project:

**Causal Learning Analytics for Early Dropout and Failure Risk Detection in Online Education**

The project develops a **stable causal-temporal learning analytics framework** for online education.  
It uses the **Open University Learning Analytics Dataset (OULAD)** to predict dropout risk, academic failure risk, and combined any-risk status from leakage-controlled weekly learner-course snapshots.

The current repository is organized around **one integrated Kaggle notebook**:

```text
notebooks/ESWA_Causal_Learning_Analytics_All_RQs_Kaggle.ipynb
```

Earlier one-notebook-per-research-question files are preserved under `legacy_notebooks/` for traceability, but the integrated notebook is the current source of record for reproducing the journal results.

---

## Repository Structure

```text
causal-learning-analytics-thesis/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── Open University Learning Analytics Dataset Link.txt
│
├── notebooks/
│   └── ESWA_Causal_Learning_Analytics_All_RQs_Kaggle.ipynb
│
├── legacy_notebooks/
│   ├── RQ1_causal_temporal_vs_baselines.ipynb
│   ├── RQ2_behavioral_casual_pathways.ipynb
│   ├── RQ3_causal_regularization_robustness.ipynb
│   ├── RQ4_counterfactual_explanations.ipynb
│   ├── RQ5_earliest_actionable_detection.ipynb
│   ├── RQ6_joint_vs_single_outcome_models.ipynb
│   └── RQ7_decision_support_pipeline.ipynb
│
├── src/
│   ├── causal_utils.py
│   ├── config.py
│   ├── counterfactual_utils.py
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── plotting_utils.py
│
├── figures/
│   └── publication-ready PDF figures generated from the integrated notebook
│
└── tables/
    └── CSV result tables generated from the integrated notebook
```

---

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

Official dataset page:

```text
https://analyse.kmi.open.ac.uk/open-dataset
```

OULAD contains learner-course records, virtual learning environment interaction logs, assessment submissions, course information, and student registration information.  
In this project, the raw OULAD records are transformed into **cumulative weekly learner-course snapshots** at Weeks 2, 4, 6, 8, 10, and 12.

The main prediction targets are:

- **Dropout risk**: final result is `Withdrawn`
- **Academic failure risk**: final result is `Fail`
- **Any-risk status**: dropout or academic failure

To reduce data leakage, the experiments use a learner-course-level split, so the same learner-course record does not appear in both training and evaluation sets.

---

## Main Methodology

The framework combines prediction, calibration, behavioral-effect estimation, counterfactual recourse, robustness evaluation, and advisor-oriented triage.

The integrated notebook includes the following analysis modules:

1. **Early risk prediction**
   - Compares Logistic Regression, Random Forest, XGBoost, HistGradientBoosting, and the stable feature model.
   - Evaluates weekly dropout and failure prediction across Weeks 2, 4, 6, 8, 10, and 12.

2. **Temporally ordered behavioral-effect evidence**
   - Estimates adjusted behavior-outcome relationships under observed-confounder assumptions.
   - Uses doubly robust estimation and bootstrap sign-stability analysis.

3. **Cross-course transfer and synthetic shift robustness**
   - Tests transfer behavior across OULAD course modules.
   - Evaluates synthetic behavioral distribution shift sensitivity.

4. **Counterfactual recourse**
   - Searches for feasible behavioral changes that reduce model-predicted dropout or failure risk.
   - Evaluates validity, plausibility, sparsity, effort, and predicted risk reduction.

5. **Early intervention timing and utility**
   - Compares early-warning usefulness across weekly observation windows.
   - Evaluates threshold and utility sensitivity under different intervention cost assumptions.

6. **Joint dropout-failure risk modeling**
   - Compares separate and joint outcome modeling.
   - Builds an intervention taxonomy using dropout-failure risk space.

7. **Capacity-aware advisor triage**
   - Compares probability-only ranking with causal-actionable ranking.
   - Evaluates workload, precision, recall, expected intervention benefit, and subgroup alert precision.

---

## Key Results

The main findings are summarized below.

### Early Risk Prediction

The stable model achieved competitive discrimination for dropout and failure prediction while improving calibration reliability.

- Dropout AUROC at Week 12: approximately **0.893**
- Failure AUROC at Week 12: approximately **0.764**
- Dropout ECE at Week 12: approximately **0.012**
- Failure ECE at Week 12: approximately **0.012**

Failure prediction was sensitive to the decision threshold.  
After out-of-fold threshold optimization, the failure F1-score improved from approximately **0.226** to approximately **0.498**.

### Behavioral-Effect Evidence

Temporally ordered doubly robust estimates showed that activity continuation, assessment performance, VLE participation, and submission behavior are strongly associated with dropout and failure risk.  
The behavioral-effect graph is interpreted as **stable adjusted evidence under observed-confounder assumptions**, not as definitive causal proof.

### Transfer and Robustness

Cross-course transfer showed mixed AUROC gains, but calibration improved more consistently.  
This supports interpreting the stable feature strategy as a **reliability-oriented modeling approach** rather than a universal accuracy-maximization method.

### Counterfactual Recourse

The counterfactual module generated feasible behavior-change suggestions for many high-risk learners.

- Dropout recourse validity: approximately **0.60**
- Failure recourse validity: approximately **0.92**
- Mean absolute risk reduction for dropout: approximately **0.198**
- Mean absolute risk reduction for failure: approximately **0.255**

These results should be interpreted as **model-space recourse**, not as confirmed real-world intervention effects.

### Early Intervention Utility

Earlier weeks provide more time for intervention, while later weeks provide stronger discrimination.  
The empirical utility analysis showed that Week 4 provided a strong balance between early lead time and predictive usefulness, while threshold-based utility optimization may prefer later weeks depending on cost assumptions.

### Joint Outcome Modeling

Joint dropout-failure modeling did not provide a large predictive advantage over separate models, but it produced a useful risk-space taxonomy for intervention planning.

The joint risk segments support different institutional strategies:

- High dropout / high failure: high-priority advising and academic support
- High dropout / low failure: engagement and retention support
- Low dropout / high failure: tutoring and assessment support
- Low dropout / low failure: routine monitoring

### Advisor Triage

Capacity-aware advisor triage showed that probability-only ranking can maximize precision when institutional capacity is very limited.  
However, causal-actionable ranking can increase expected intervention benefit by prioritizing students who are both high-risk and more likely to benefit from feasible behavioral recourse.

---

## Figures

The `figures/` folder contains publication-ready PDF figures, including:

```text
figure_1_1_early_week_auroc_dropout.pdf
figure_1_1_early_week_auroc_failure.pdf
figure_1_2_failure_f1_default_vs_optimized_threshold.pdf
figure_2_1_dr_effect_forest_dropout.pdf
figure_2_1_dr_effect_forest_failure.pdf
figure_2_2_stable_temporal_behavior_graph.pdf
figure_3_1_cross_course_transfer_AUROC_delta_dropout.pdf
figure_3_1_cross_course_transfer_AUROC_delta_failure.pdf
figure_3_1_cross_course_transfer_ECE_delta_dropout.pdf
figure_3_1_cross_course_transfer_ECE_delta_failure.pdf
figure_3_2_shift_sensitivity_dropout.pdf
figure_3_2_shift_sensitivity_failure.pdf
figure_4_1_counterfactual_effort_benefit_frontier.pdf
figure_4_2_observed_vs_counterfactual_risk.pdf
figure_5_1_early_warning_empirical_utility.pdf
figure_5_2_threshold_net_utility_surface.pdf
figure_5_3_utility_sensitivity_by_week.pdf
figure_6_1_joint_dropout_failure_risk_space.pdf
figure_6_2_calibration_curve_dropout.pdf
figure_6_2_calibration_curve_failure.pdf
figure_6_3_shared_vs_specific_contributions.pdf
figure_7_1_workload_precision_recall_expected_benefit.pdf
figure_7_2_institutional_decision_support_pipeline.pdf
figure_7_3_real_subgroup_alert_precision_audit.pdf
```

---

## Tables

The `tables/` folder contains CSV tables generated by the integrated notebook.  
These include weekly prediction metrics, behavioral-effect estimates, transfer results, counterfactual quality metrics, intervention utility simulations, joint risk-space summaries, advisor triage outputs, subgroup audits, and an output manifest.

Examples include:

```text
table_1_2_weekly_model_comparison_summary.csv
table_2_1_doubly_robust_temporal_behavioral_effects.csv
table_3_1_cross_course_transfer.csv
table_3_3_robustness_summary.csv
table_4_2_counterfactual_quality_metrics.csv
table_5_1_weekly_intervention_readiness.csv
table_6_1_joint_vs_separate_outcome_metrics.csv
table_7_1_capacity_constrained_ground_truth_triage.csv
table_7_2_real_subgroup_fairness_actionability_audit.csv
table_8_2_output_manifest.csv
```

---

## How to Run

### Option 1: Run on Kaggle

1. Create a new Kaggle Notebook.
2. Upload the OULAD dataset as a Kaggle input dataset.
3. Upload this repository or copy the repository contents into the working directory.
4. Open:

```text
notebooks/ESWA_Causal_Learning_Analytics_All_RQs_Kaggle.ipynb
```

5. Set the dataset path in the notebook if needed.
6. Run all cells.

The notebook saves figures as PDF files and tables as CSV files.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/Seoyeon2Kim/causal-learning-analytics-thesis.git
cd causal-learning-analytics-thesis
```

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Open the integrated notebook:

```bash
jupyter notebook notebooks/ESWA_Causal_Learning_Analytics_All_RQs_Kaggle.ipynb
```

Place the OULAD CSV files in a local data directory or update the dataset path inside the notebook.

---

## Reproducibility Notes

- The current journal results are generated from the integrated notebook in `notebooks/`.
- The `legacy_notebooks/` folder is retained only for historical traceability.
- The repository no longer depends on EdNet, MOOCCube, or MOOCCubeX.
- All reported robustness tests are based on **cross-course transfer within OULAD** and **synthetic behavioral distribution shift**.
- Causal language should be interpreted carefully: behavioral-effect estimates are based on temporal ordering and observed-confounder adjustment, not randomized experiments.
- Counterfactual recourse is evaluated in model space and requires institutional validation before real-world deployment.

---

## Citation

If you use this repository, please cite the associated thesis or journal manuscript:

```text
Kim, S., Ali, R. H., & Khan, T. A.
Causal Learning Analytics for Early Dropout and Failure Risk Detection in Online Education.
Manuscript in preparation / thesis project, University of Europe for Applied Sciences.
```

---

## Author

**Seoyeon Kim**  
University of Europe for Applied Sciences  
MSc Data Science / Learning Analytics
