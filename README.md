# Kaggle Blueprints

An experimental tabular-classification notebook with data exploration, preprocessing, boosted-tree models, hyperparameter search, feature analysis, and ensembling. The checked-in example uses Titanic passenger data.

**[Open the notebook](notebooks/Tabular%20Data%20Notebook.ipynb)** · **[Sample inputs](input/)** · **[MIT license](LICENSE)**

## Contents

- [Repository guide](#repository-guide)
- [Working with the notebook](#working-with-the-notebook)
- [Environment](#environment)
- [Preprocessing and generated files](#preprocessing-and-generated-files)
- [Saved notebook results](#saved-notebook-results)
- [Focused regression checks](#focused-regression-checks)
- [License](#license)

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

Saved notebook metadata records Python 3.10.9. The header imports NumPy, pandas, scikit-learn, SciPy, matplotlib, Featuretools, fastai, DataPrep, MLflow, Optuna, LightGBM, XGBoost, CatBoost, TPOT, and SHAP. The tuning cells use the `jupyternotify` extension (`%load_ext jupyternotify`, `%%notify`), and the MLflow cells log to a tracking server at `http://127.0.0.1:5000/`.

The Autokeras section is an unfinished regression template: it references AutoKeras, Keras callbacks, `KFold`, `tqdm`, and `mean_squared_error`, which the notebook does not import, and variables that the classification workflow does not create.

The existing requirements file is a broad, older environment export with platform-specific packages. It is not a verified portable installation recipe or a complete match for every optional experiment. Build an isolated environment for the sections you intend to run and record the versions used.

## Preprocessing and generated files

Category mappings, median imputation, and scaling are fitted on training features and then applied to validation and test features. Unseen categorical values use -1, and missing numeric features use training medians. DataFrame columns and row indices are preserved.

The earlier imputation and polynomial sections define optional helpers. They can be read in notebook order without accessing training variables before the split exists. Call those helpers only after preparing the appropriate numeric splits; the default workflow keeps its original feature set.

The export cell writes features and targets under `output/processed/`, preserving the original files in `input/`:

- `train_features.csv` and `train_targets.csv`
- `validation_features.csv` and `validation_targets.csv`
- `test_features.csv`

These files include a `row_id` column for alignment. Model sections may also write experiment logs, TPOT exports, a DataPrep report under `Reports/`, and submission CSVs with the `PassengerId,Survived` columns of `input/gender_submission.csv`. Apart from the tracked TPOT example, these generated files are listed in `.gitignore`. Kernel shutdown is optional and disabled by default.

## Saved notebook results

Copied from the outputs saved in the notebook (Titanic data, March 2023). The cells ran across several sessions and some outputs were printed by earlier versions of the code, so treat them as a historical record, not a benchmark.

| Notebook step | Saved output |
| --- | --- |
| Adversarial validation, numeric columns with `PassengerId` | test AUC 0.999 |
| Adversarial validation, after dropping `PassengerId` | test AUC 0.531 |
| LightGBM, 5-fold CV accuracy on the training split | 0.799 |
| XGBoost, 5-fold CV accuracy on the training split | 0.803 |
| CatBoost, 5-fold CV accuracy on the training split | 0.817 |
| TPOT, best internal CV accuracy | 0.828 |
| TPOT, accuracy on the validation split | 0.810 |

The adversarial-validation drop shows `PassengerId` alone separated the training and test files. The saved hyperparameter-search outputs are omitted because their logged parameters do not match the search spaces in the current cells.

## Focused regression checks

```bash
python -m pip install -r requirements-checks.txt
python -m unittest discover -s tests -v
```

The checks cover every code cell parsing, the submission schema, required imports, the tuning-cell typo fixes, SHAP plot inputs, the MLflow setup order, preprocessing/export behavior, unseen and missing categorical values, training-only imputation, preprocessing cell execution order, optional feature recipes, and logger timestamp formatting. The pinned check environment uses Python 3.12. These tests use pandas and scikit-learn without executing the complete notebook or training models.

Saved notebook outputs are historical records. The full experiment environment and model scores have not been reproduced by these focused checks.

## License

[MIT](LICENSE).
