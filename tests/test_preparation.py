"""Exercise the preparation cell without notebook integrations or model training."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def prepare(validation_age=1000, categorical=False):
    path = Path(__file__).resolve().parents[1] / "notebooks/Tabular Data Notebook.ipynb"
    cells = json.loads(path.read_text())["cells"]
    code = next("".join(cell["source"]) for cell in cells
                if "category_encoder = OrdinalEncoder" in "".join(cell["source"]))
    namespace = {"pd": pd, "np": np, "SimpleImputer": SimpleImputer,
                 "cat_cols": ["Sex"], "num_cols": ["Age"],
                 "X_train": pd.DataFrame({"Sex": ["male", "female", "male"],
                                          "Age": [10.0, np.nan, 30.0]}, index=[2, 4, 8]),
                 "X_val": pd.DataFrame({"Sex": ["unseen", None], "Age": [validation_age, np.nan]}, index=[7, 10]),
                 "X_test": pd.DataFrame({"Sex": [None], "Age": [np.nan]}, index=[21])}
    if categorical:
        for name in ("X_train", "X_val", "X_test"):
            namespace[name]["Sex"] = namespace[name]["Sex"].astype("category")
    exec(code, namespace)
    return namespace


class PreparationTests(unittest.TestCase):
    def test_categorical_dtype_accepts_missing_values_without_new_category_error(self):
        ns = prepare(categorical=True)
        self.assertEqual(ns["X_val"]["Sex"].tolist(), [-1.0, -1.0])
        self.assertEqual(ns["X_test"]["Sex"].tolist(), [-1.0])
        self.assertFalse(ns["X_train"].isna().any().any())

    def test_unseen_and_missing_categories_are_encodable(self):
        ns = prepare()
        self.assertEqual(ns["X_val"]["Sex"].tolist(), [-1.0, -1.0])
        self.assertEqual(ns["X_test"]["Sex"].tolist(), [-1.0])
        self.assertFalse(ns["X_train"].isna().any().any())
        self.assertFalse(ns["X_val"].isna().any().any())

    def test_holdout_values_do_not_change_training_imputation(self):
        first, changed = prepare(1000), prepare(-1000)
        pd.testing.assert_frame_equal(first["X_train"], changed["X_train"])
        self.assertEqual(first["X_train"].loc[4, "Age"], 20.0)
        self.assertEqual(first["X_val"].loc[10, "Age"], 20.0)
        self.assertEqual(first["X_test"].loc[21, "Age"], 20.0)
        self.assertEqual(first["X_train"].index.tolist(), [2, 4, 8])

    def test_eda_uses_loaded_source_before_memory_optimization_cell(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/Tabular Data Notebook.ipynb"
        cells = json.loads(path.read_text())["cells"]
        code = next("".join(cell["source"]) for cell in cells if "report = create_report(" in "".join(cell["source"]))
        sentinel, captured = object(), []
        class Report:
            def save(self, path): captured.append(path)
        def create_report(frame, **kwargs):
            self.assertIs(frame, sentinel)
            return Report()
        exec(code, {"train_org": sentinel, "create_report": create_report})
        self.assertEqual(len(captured), 1)

    def test_optional_recipes_can_be_defined_before_training_splits_exist(self):
        cells = json.loads((Path(__file__).resolve().parents[1] /
                            "notebooks/Tabular Data Notebook.ipynb").read_text())["cells"]
        namespace = {"pd": pd, "SimpleImputer": SimpleImputer,
                     "PolynomialFeatures": PolynomialFeatures}
        for cell in cells:
            code = "".join(cell.get("source", []))
            if code.startswith("# Optional recipe"):
                exec(code, namespace)
        self.assertNotIn("X_train", namespace)
        train = pd.DataFrame({"age": [1.0, np.nan, 3.0]}, index=[2, 4, 8])
        validation = pd.DataFrame({"age": [100.0, np.nan]}, index=[7, 10])
        original = train.copy()
        prepared_train, prepared_validation = namespace["impute_numeric_splits"](train, validation)
        self.assertEqual(prepared_validation["age"].tolist(), [100.0, 2.0])
        self.assertEqual(prepared_train.index.tolist(), [2, 4, 8])
        expanded_train, expanded_validation = namespace["polynomial_feature_splits"](
            prepared_train, prepared_validation, degree=2)
        self.assertEqual(expanded_validation["age^2"].tolist(), [10000.0, 4.0])
        self.assertEqual(expanded_train.columns.tolist(), expanded_validation.columns.tolist())
        pd.testing.assert_frame_equal(train, original)

    def test_preparation_cells_run_in_notebook_order_with_categorical_inputs(self):
        cells = json.loads((Path(__file__).resolve().parents[1] /
                            "notebooks/Tabular Data Notebook.ipynb").read_text())["cells"]
        train = pd.DataFrame({
            "PassengerId": range(1, 13), "Name": ["Person"] * 12,
            "Ticket": ["T"] * 12, "Cabin": [None] * 12,
            "Survived": [0, 1] * 6,
            "Sex": pd.Categorical(["female", "male", None] * 4),
            "Age": [10.0, 20.0, np.nan, 40.0] * 3,
            "Fare": [1.0, 2.0, 3.0] * 4,
        })
        test = train.drop(columns="Survived").iloc[:2].copy()
        test.index = [40, 41]
        test["Sex"] = pd.Categorical(["unseen", None])
        test["Age"] = [1000.0, np.nan]
        original_train, original_test = train.copy(), test.copy()
        namespace = {"train": train, "test": test, "pd": pd, "np": np,
                     "SimpleImputer": SimpleImputer, "PolynomialFeatures": PolynomialFeatures,
                     "MinMaxScaler": MinMaxScaler, "train_test_split": train_test_split,
                     "C": SimpleNamespace(RANDOM_SEED=42)}
        for index in (31, 33, 35, 36, 37, 38, 39):
            exec("".join(cells[index]["source"]), namespace)
        for name in ("X_train", "X_val", "X_test"):
            self.assertTrue(np.isfinite(namespace[name].to_numpy()).all(), name)
            self.assertEqual(namespace[name].columns.tolist(), ["Sex", "Age", "Fare"])
        self.assertEqual(namespace["X_test"].index.tolist(), [40, 41])
        self.assertEqual(namespace["X_train"].index.tolist(), namespace["y_train"].index.tolist())
        self.assertEqual(namespace["X_val"].index.tolist(), namespace["y_val"].index.tolist())
        pd.testing.assert_frame_equal(train, original_train)
        pd.testing.assert_frame_equal(test, original_test)


if __name__ == "__main__":
    unittest.main()
