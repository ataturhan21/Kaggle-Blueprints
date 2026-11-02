# Kaggle Blueprints

An experimental tabular-classification notebook with data exploration, preprocessing, boosted-tree models, hyperparameter search, feature analysis, and ensembling. The checked-in example uses Titanic passenger data.

**[Open the notebook](notebooks/Tabular%20Data%20Notebook.ipynb)** · **[Sample inputs](input/)** · **[MIT license](LICENSE)**

## Repository guide

| Path | Purpose |
| --- | --- |
| `notebooks/Tabular Data Notebook.ipynb` | Main experimental workflow |
| `input/` | Source training/test CSVs and a sample submission |
| `notebooks/tpot_pipeline.py` | Exported TPOT example with placeholder data path and separator |
| `logger/` | Console and optional file logging |
| `requirements.txt` | Historical development-environment export |

## Working with the notebook

```bash
git clone https://github.com/ataturhan21/Kaggle-Blueprints.git
cd Kaggle-Blueprints
```

Open the notebook with its kernel working directory set to `notebooks/`. Its relative paths expect the input CSVs and logger directory 1 level above it.

1. Review the imports and the `C` configuration class.
2. Inspect the input data and target-specific transformations.
3. Run the preparation cells before selecting model or hyperparameter-search sections.
4. Review feature alignment and the required submission schema before exporting predictions.

This is a research blueprint with experiment-specific cells, not a packaged command-line training pipeline. The TPOT export requires replacing `PATH/TO/DATA/FILE`, `COLUMN_SEPARATOR`, and the assumed target column before use.

## Environment

Saved notebook metadata records Python 3.10.9. The header imports NumPy, pandas, scikit-learn, SciPy, matplotlib, Featuretools, fastai, DataPrep, MLflow, Optuna, LightGBM, XGBoost, CatBoost, and TPOT.

The existing requirements file is a broad, older environment export with platform-specific packages. It is not a verified portable installation recipe or a complete match for every optional experiment. Build an isolated environment for the sections you intend to run and record the versions used.

## Preprocessing and generated files

Category mappings, median imputation, and scaling are fitted on training features and then applied to validation and test features. Unseen categorical values use -1, and missing numeric features use training medians. DataFrame columns and row indices are preserved.

The earlier imputation and polynomial sections define optional helpers. They can be read in notebook order without accessing training variables before the split exists. Call those helpers only after preparing the appropriate numeric splits; the default workflow keeps its original feature set.

The export cell writes features and targets under `output/processed/`, preserving the original files in `input/`:

- `train_features.csv` and `train_targets.csv`
- `validation_features.csv` and `validation_targets.csv`
- `test_features.csv`

These files include a `row_id` column for alignment. Model sections may also write experiment logs, TPOT exports, and submission CSVs. Kernel shutdown is optional and disabled by default.

## Focused regression checks

```bash
python -m pip install -r requirements-checks.txt
python -m unittest discover -s tests -v
```

The checks cover preprocessing/export behavior, unseen and missing categorical values, training-only imputation, preprocessing cell execution order, optional feature recipes, and logger timestamp formatting. The pinned check environment uses Python 3.12. These tests use pandas and scikit-learn without executing the complete notebook or training models.

Saved notebook outputs are historical records. The full experiment environment and model scores have not been reproduced by these focused checks.

## License

[MIT](LICENSE).
