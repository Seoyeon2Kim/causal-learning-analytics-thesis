# Early Dropout and Failure Risk Detection in Online Education — Kaggle Code Bundle

This package converts the thesis proposal into a runnable Kaggle workflow with **one standalone notebook per research question**.

## Contents

- `notebooks/RQ1_causal_temporal_vs_baselines.ipynb`
- `notebooks/RQ2_behavioral_causal_pathways.ipynb`
- `notebooks/RQ3_causal_regularization_robustness.ipynb`
- `notebooks/RQ4_counterfactual_explanations.ipynb`
- `notebooks/RQ5_earliest_actionable_detection.ipynb`
- `notebooks/RQ6_joint_vs_single_outcome_models.ipynb`
- `notebooks/RQ7_decision_support_pipeline.ipynb`
- `src/*.py` shared utilities
- `requirements.txt`

## How to use on Kaggle

1. Create a new Kaggle notebook.
2. Add your educational dataset under **Add Input**.
3. Upload this code bundle as a Kaggle dataset or copy the folder into `/kaggle/working/seoyeon_thesis_code`.
4. Open the required RQ notebook.
5. Update `DATASET_PATH` if needed.
6. Run all cells.

## Outputs

Each notebook writes files to:

`/kaggle/working/results/<rq_notebook_name>/`

Outputs include:

- publication-ready PDF figures
- CSV tables
- intermediate evaluation summaries

## Notes

- The data loader is schema-flexible and tries to normalize event-log style datasets such as MOOCCube, MOOCCubeX, and EdNet into a common weekly timeline.
- Where those datasets expose different field names, the loader maps common aliases automatically.
- Some proposal items, especially causal graph discovery and counterfactual recourse, are implemented as practical, reproducible approximations suitable for a thesis prototype and for Kaggle execution.
- If you have cleaned dataset-specific schemas already, you can replace `load_generic_education_dataset()` with a stricter dataset adapter while keeping the rest of the pipeline unchanged.

## Expected artifacts by notebook

- **RQ1:** weekly benchmark comparison, early-risk AUROC curves, accuracy–actionability trade-off, component ablation tables
- **RQ2:** learned causal edge tables, pathway decomposition plots, direct/indirect effect tables
- **RQ3:** cross-domain transfer matrices, domain-shift heatmaps, robustness decay curves
- **RQ4:** counterfactual recourse examples, counterfactual quality table, effort–benefit frontier, risk-trajectory plots
- **RQ5:** intervention-readiness tables, utility-over-time curves, adaptive threshold surfaces, workload simulation tables
- **RQ6:** joint-vs-separate model comparison, joint risk-space plot, shared/task-specific contribution analysis
- **RQ7:** institutional pipeline figure, capacity-constrained triage table, workload–benefit decision curve, fairness/actionability audit
