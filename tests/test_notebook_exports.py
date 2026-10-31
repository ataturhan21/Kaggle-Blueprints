import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


NOTEBOOK = json.loads(
    (Path(__file__).resolve().parents[1] / "notebooks/Tabular Data Notebook.ipynb").read_text()
)


def cell_containing(text):
    for cell in NOTEBOOK["cells"]:
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if text in source:
            return source
    raise AssertionError(f"Notebook cell not found: {text}")


def inputs():
    return {
        "pd": pd,
        "os": os,
        "MinMaxScaler": MinMaxScaler,
        "X_train": pd.DataFrame({"age": [0.0, 10.0], "fare": [2.0, 4.0]}, index=[10, 20]),
        "X_val": pd.DataFrame({"age": [5.0], "fare": [3.0]}, index=[30]),
        "X_test": pd.DataFrame({"age": [20.0], "fare": [6.0]}, index=[40]),
        "y_train": pd.Series([0, 1], index=[10, 20], name="Survived"),
        "y_val": pd.Series([1], index=[30], name="Survived"),
    }


class ExportTests(unittest.TestCase):
    def test_preparation_uses_unlabeled_test_rows_instead_of_training_rows(self):
        train = pd.DataFrame({
            "PassengerId": [1, 2], "Name": ["train A", "train B"],
            "Ticket": ["a", "b"], "Cabin": ["A1", "B1"],
            "Age": [20.0, 30.0], "Fare": [10.0, 40.0], "Survived": [0, 1],
        }, index=[10, 20])
        test = pd.DataFrame({
            "PassengerId": [901, 902], "Name": ["test A", "test B"],
            "Ticket": ["c", "d"], "Cabin": ["C1", "D1"],
            "Age": [50.0, 60.0], "Fare": [80.0, 90.0],
        }, index=[90, 91])
        namespace = {"train_org": train, "test_org": test,
                     "df_shrink": lambda frame, **kwargs: frame.copy()}
        for index in [22, 23, 35]:
            exec("".join(NOTEBOOK["cells"][index]["source"]), namespace)
        self.assertEqual(namespace["test"]["PassengerId"].tolist(), [901, 902])
        pd.testing.assert_frame_equal(namespace["X_test"], test[["Age", "Fare"]])
        self.assertNotIn("Survived", namespace["X_test"].columns)
        self.assertEqual(namespace["y_train_org"].tolist(), [0, 1])

    def test_scaling_keeps_dataframes_and_applies_training_scale_to_test_data(self):
        namespace = inputs()
        exec(cell_containing("scaler = MinMaxScaler()"), namespace)
        expected = pd.DataFrame({"age": [2.0], "fare": [2.0]}, index=[40])
        pd.testing.assert_frame_equal(namespace["X_test"], expected)
        self.assertEqual(namespace["X_train"].index.tolist(), [10, 20])
        self.assertEqual(namespace["X_train"].columns.tolist(), ["age", "fare"])

    def test_export_preserves_raw_inputs_and_feature_target_alignment(self):
        namespace = inputs()
        exec(cell_containing("scaler = MinMaxScaler()"), namespace)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "notebooks").mkdir()
            (root / "input/train.csv").write_text("raw training input\n")
            (root / "input/test.csv").write_text("raw test input\n")
            try:
                os.chdir(root / "notebooks")
                exec("".join(NOTEBOOK["cells"][41]["source"]), namespace)
            finally:
                os.chdir(previous)
            self.assertEqual((root / "input/train.csv").read_text(), "raw training input\n")
            self.assertEqual((root / "input/test.csv").read_text(), "raw test input\n")
            output = root / "output/processed"
            self.assertEqual(len(list(output.glob("*.csv"))), 5)
            features = pd.read_csv(output / "train_features.csv")
            targets = pd.read_csv(output / "train_targets.csv")
            self.assertEqual(features["row_id"].tolist(), [10, 20])
            self.assertEqual(targets["row_id"].tolist(), [10, 20])
            self.assertEqual(targets["Survived"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
