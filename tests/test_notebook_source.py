"""Static checks on the notebook source; nothing is executed or trained."""
import ast
import csv
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Tabular Data Notebook.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).splitlines()
        # IPython magics (%, %%) and shell (!) lines are not Python syntax.
        cells.append("\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in lines))
    return cells


def cell_with(cells, text):
    return next(cell for cell in cells if text in cell)


def imported_names(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return names


class NotebookSourceTests(unittest.TestCase):
    def setUp(self):
        self.cells = code_cells()
        self.imports = imported_names(self.cells[0])

    def test_every_code_cell_parses(self):
        for index, cell in enumerate(self.cells):
            with self.subTest(code_cell=index):
                compile(cell, f"code-cell-{index}", "exec")

    def test_names_used_by_scoring_and_explanations_are_imported(self):
        self.assertIn("accuracy_score", self.imports)
        self.assertIn("shap", self.imports)

    def test_tuning_typos_stay_fixed(self):
        source = "\n".join(self.cells)
        self.assertNotIn('params["n_estimators"]', cell_with(self.cells, "CatBoostPruningCallback(trial"))
        self.assertNotIn("C.EVAL_METRIC **", source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                self.assertNotIn("EVAL_METRIC", ast.unparse(node.left))
            if isinstance(node, ast.Assign) and "num_leaves" in ast.unparse(node.targets[0]):
                self.assertNotIsInstance(node.value, ast.Tuple)

    def test_shap_plots_use_the_explained_frame(self):
        for cell in self.cells:
            if "shap_values(X)" in cell:
                self.assertNotIn("summary_plot(shap_values, X_train", cell)

    def test_tracking_uri_is_set_before_experiment_lookup(self):
        cell = cell_with(self.cells, "create_experiment")
        self.assertLess(cell.index("set_tracking_uri"), cell.index("create_experiment"))

    def test_submission_matches_the_sample_submission_schema(self):
        code = cell_with(self.cells, 'submission.to_csv("submission.csv"')
        written = {}

        class Model:
            def predict(self, frame):
                return np.zeros(len(frame), dtype=int)

        class Frame(pd.DataFrame):
            def to_csv(self, path, **kwargs):
                written[path] = pd.DataFrame(self).to_csv(**kwargs)

        test = pd.DataFrame({"PassengerId": [892, 893], "Age": [30.0, 40.0]}, index=[5, 6])
        namespace = {"pd": type("pd", (), {"DataFrame": Frame}), "clf": Model(),
                     "test": test, "X_test": test[["Age"]],
                     "time": type("time", (), {"strftime": staticmethod(lambda fmt: "now")})}
        exec(code, namespace)
        with open(ROOT / "input/gender_submission.csv", newline="") as sample:
            expected_header = next(csv.reader(sample))
        self.assertEqual(written["submission.csv"].splitlines()[0].split(","), expected_header)
        self.assertEqual(written["submission.csv"].splitlines()[1:], ["892,0", "893,0"])


if __name__ == "__main__":
    unittest.main()
