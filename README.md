# Early Dropout and Failure Risk Detection in Online Education  
### Reproducible Kaggle Workflow for RQ1–RQ7 (Thesis Code Bundle)

This repository contains the full reproducible codebase for the thesis:  
**“Causal Learning Analytics for Early Dropout and Failure Risk Detection in Online Education.”**

The project is organized as a **Kaggle‑ready workflow**, with **one notebook per research question (RQ1–RQ7)**,  
plus shared utilities, figures, tables, and dataset adapters.

All experiments—including causal graph discovery, robustness evaluation, counterfactual recourse,  
and institutional decision‑support simulation—can be reproduced end‑to‑end.

---

# 📁 Repository Structure

```
seoyeon-thesis/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── EdNet_question_features.csv
│
├── figures/                # Publication-ready PDF figures
│   ├── figure_1_1_early_risk_performance_over_time.pdf
│   ├── figure_1_2_accuracy_actionability_tradeoff.pdf
│   ├── figure_2_1_learned_temporal_causal_graph.pdf
│   ├── figure_2_2_direct_and_mediated_pathway_decomposition.pdf
│   ├── figure_3_1_domain_shift_sensitivity_heatmap.pdf
│   ├── figure_3_2_performance_decay_under_synthetic_shift_severity.pdf
│   ├── figure_4_1_effort_benefit_frontier.pdf
│   ├── figure_4_2_observed_vs_counterfactual_risk_trajectories.pdf
│   ├── figure_5_1_early_warning_utility_over_course_time.pdf
│   ├── figure_5_2_threshold_surface_for_adaptive_alerting.pdf
│   ├── figure_6_1_joint_dropout_failure_risk_space.pdf
│   ├── figure_6_2_shared_vs_task_specific_behavioral_contributions.pdf
│   ├── figure_7_1_institutional_decision_support_pipeline.pdf
│   └── figure_7_2_workload_benefit_decision_curve.pdf
│
├── tables/                 # All CSV tables used in the thesis
│   ├── table_1_1_weekly_benchmark_comparison.csv
│   ├── table_1_2_accuracy_actionability_points.csv
│   ├── table_1_3_component_ablation_analysis.csv
│   ├── table_2_1_stable_causal_edges_and_effect_strengths.csv
│   ├── table_2_2_direct_indirect_total_effects_dropout.csv
│   ├── table_2_3_direct_indirect_total_effects_failure.csv
│   ├── table_2_4_combined_behavior_effects.csv
│   ├── table_3_1_cross_dataset_transfer_performance_matrix.csv
│   ├── table_3_2_synthetic_shift_severity_results.csv
│   ├── table_3_3_stability_metrics_across_domains.csv
│   ├── table_3_4_ednet_cross_platform_robustness.csv
│   ├── table_4_1_counterfactual_recourse_examples.csv
│   ├── table_4_2_counterfactual_quality_evaluation.csv
│   ├── table_4_3_counterfactual_risk_trajectories.csv
│   ├── table_5_1_weekly_intervention_readiness_metrics.csv
│   ├── table_5_2_intervention_workload_simulation_by_week_threshold.csv
│   ├── table_6_1_joint_vs_separate_model_comparison.csv
│   ├── table_6_2_joint_dropout_failure_risk_space.csv
│   ├── table_6_3_shared_vs_task_specific_behavioral_contributions.csv
│   ├── table_6_4_intervention_strategies_by_joint_risk_segment.csv
│   ├── table_7_1_capacity_constrained_triage_outcomes.csv
│   ├── table_7_2_workload_benefit_curve.csv
│   └── table_7_3_fairness_and_actionability_audit.csv
│
├── notebooks/              # One notebook per research question
│   ├── RQ1_causal_temporal_vs_baselines.ipynb
│   ├── RQ2_behavioral_casual_pathways.ipynb
│   ├── RQ3_causal_regularization_robustness.ipynb
│   ├── RQ4_counterfactual_explanations.ipynb
│   ├── RQ5_earliest_actionable_detection.ipynb
│   ├── RQ6_joint_vs_single_outcome_models.ipynb
│   └── RQ7_decision_support_pipeline.ipynb
│
└── src/                    # Shared utilities for all notebooks
├── init.py
├── causal_utils.py
├── config.py
├── counterfactual_utils.py
├── data_utils.py
├── evaluation.py
├── feature_engineering.py
├── models.py
└── plotting_utils.py
```

---


# 🚀 How to Run on Kaggle

1. Create a new Kaggle Notebook.
2. Upload your dataset under **Add Input**  
   (supports OULAD, MOOCCube, MOOCCubeX, EdNet, or any event‑log dataset).
3. Upload this repository as a Kaggle Dataset  
   or copy the folder into: `/kaggle/working/seoyeon_thesis_code/`

4. Open any RQ notebook (e.g., `RQ3_causal_regularization_robustness.ipynb`).
5. Set the dataset path:
```python
DATASET_PATH = "/kaggle/input/<your-dataset>"
```
6. Run all cells.
All outputs will be saved to:
```/kaggle/working/results/<rq_name>/```

# 📊 Outputs by Notebook
## RQ1 — Causal Temporal Model vs Baselines
- Weekly AUROC curves
- Accuracy–actionability trade-off
- Component ablation tables
- Early-warning performance plots

## RQ2 — Behavioral Causal Pathways
- Learned temporal causal graph
- Stable causal edges table
- Direct/indirect/total effect decomposition
- Mediated pathway visualization

## RQ3 — Causal Regularization & Robustness
- Cross-dataset transfer matrix
- Domain-shift sensitivity heatmap
- Synthetic shift severity decay curve
- Stability metrics (std AUROC, explanation drift, graph consistency)
- EdNet cross-platform robustness table

## RQ4 — Counterfactual Explanations
- Counterfactual recourse examples
- Counterfactual quality evaluation
- Effort–benefit frontier
- Observed vs counterfactual risk trajectories

## RQ5 — Earliest Actionable Detection
- Weekly intervention-readiness metrics
- Utility-over-time curves
- Adaptive threshold surface
- Workload simulation tables

## RQ6 — Joint vs Single Outcome Models
- Joint dropout–failure risk space
- Shared vs task-specific behavioral contributions
- Joint vs separate model comparison
- Intervention strategies by risk segment

## RQ7 — Institutional Decision-Support Pipeline
- Full 7‑stage institutional pipeline figure
- Capacity‑constrained triage outcomes
- Workload–benefit decision curve
- Fairness & actionability audit

# 🧠 Key Features
## ✔ Schema-flexible data loader

Automatically normalizes event‑log datasets (OULAD, MOOCCube, EdNet)
into a unified weekly timeline.

## ✔ Causal graph discovery
PCMCI‑style temporal causal discovery with stability filtering.

## ✔ Causal regularization
Improves domain robustness and reduces explanation drift.

## ✔ Counterfactual recourse
Generates realistic, feasible behavioral recommendations.

## ✔ Cross-platform generalization
Includes EdNet question‑feature alignment for platform‑shift evaluation.

## ✔ Publication-ready outputs
All figures exported as 300+ ppi PDF, ready for journal submission.

# 🧩 Notes for Reproducibility
- All figures and tables in the thesis are generated directly from these notebooks.

- All intermediate artifacts are saved under /results/ for traceability.

- The code avoids dataset‑specific assumptions; if you have a custom schema,
replace `load_generic_education_dataset()` with your own adapter.

- Counterfactual and causal components use practical approximations
optimized for Kaggle runtime constraints.