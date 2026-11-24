"""CPU-only contract tests; imports definitions from the deliverable notebook.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -v
No raw data, network access, or real model training is required.
"""
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/vibecode-test1.ipynb"


def definitions():
    assert NOTEBOOK.exists(), "Requested notebook has not been implemented"
    namespace = {"__name__": "vibecode_contract_tests"}
    for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
        if "definitions" in cell.get("metadata", {}).get("tags", []):
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    return namespace


class NotebookContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = definitions()
        torch.set_num_threads(2)

    def test_manual_metrics_ignore_absent_and_false_positive(self):
        y = torch.tensor([0, 0, 1, 1, 255])
        p = torch.tensor([0, 1, 1, 2, 7])
        cm = self.n["confusion_counts"](p, y)
        expected = torch.zeros(8, 8, dtype=torch.int64)
        expected[0, 0] = expected[0, 1] = expected[1, 1] = expected[1, 2] = 1
        torch.testing.assert_close(cm, expected)
        table, summary = self.n["metric_report"](cm)
        np.testing.assert_allclose(table.iou[:3], [0.5, 1 / 3, 0])
        np.testing.assert_allclose(table.dice[:3], [2 / 3, 0.5, 0])
        self.assertTrue(table.iou[3:].isna().all())
        self.assertAlmostEqual(summary["miou"], 5 / 18)
        self.assertAlmostEqual(summary["pixel_accuracy"], 0.5)
        self.assertEqual(summary["classes_in_union"], 3)

    def test_global_counts_are_additive_not_image_means(self):
        fn = self.n["confusion_counts"]
        a, b = torch.tensor([0, 0]), torch.tensor([0, 1, 1])
        pa, pb = torch.tensor([0, 1]), torch.tensor([1, 1, 1])
        torch.testing.assert_close(fn(pa, a) + fn(pb, b), fn(torch.cat([pa, pb]), torch.cat([a, b])))

    def test_all_unknown_loss_is_finite_zero_and_no_gradient(self):
        logits = torch.randn(1, 8, 4, 5, requires_grad=True)
        y = torch.full((1, 4, 5), 255, dtype=torch.long)
        loss = self.n["valid_ce"](logits, y)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)
        _, summary = self.n["metric_report"](self.n["confusion_counts"](logits.argmax(1), y))
        self.assertTrue(np.isnan(summary["miou"]))

    def test_ignored_pixel_cannot_change_loss(self):
        logits = torch.randn(1, 8, 2, 2)
        y = torch.tensor([[[0, 255], [1, 2]]])
        other = logits.clone()
        other[:, :, 0, 1] = torch.arange(8) * 100
        torch.testing.assert_close(self.n["valid_ce"](logits, y), self.n["valid_ce"](other, y))

    def test_invalid_prediction_rejected(self):
        with self.assertRaises(ValueError):
            self.n["confusion_counts"](torch.tensor([8]), torch.tensor([1]))

    def test_unet_odd_shape_backward_and_checkpoint_roundtrip(self):
        model = self.n["UNet"](base=4)
        x = torch.rand(1, 3, 33, 47)
        y = torch.randint(0, 8, (1, 33, 47))
        result = model(x)
        self.assertEqual(tuple(result.shape), (1, 8, 33, 47))
        self.n["valid_ce"](result, y).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            torch.save(model.state_dict(), path)
            copy = self.n["UNet"](base=4).eval()
            copy.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), copy(x))

    def test_existing_preprocessing_preserves_geometry_ids_unknown(self):
        from PIL import Image
        from src.dataset import prepare_pair
        raw = np.zeros((8, 12), dtype=np.uint8)
        raw[:, 6:] = 8
        rgb = np.repeat((raw > 0)[..., None] * np.uint8(255), 3, axis=2)
        with tempfile.TemporaryDirectory() as folder:
            image, mask = Path(folder) / "image.png", Path(folder) / "mask.png"
            Image.fromarray(rgb).save(image)
            Image.fromarray(raw).save(mask)
            x, y = prepare_pair(image, mask)
        self.assertEqual(x.shape, (3, 512, 512))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.int64)
        self.assertEqual(set(np.unique(y)), {7, 255})
        np.testing.assert_array_equal(x[0] > 0.5, y == 7)

    def test_calibration_perfect_and_unknown(self):
        probs = torch.zeros(1, 8, 2, 2)
        probs[:, 0] = 1
        y = torch.tensor([[[0, 0], [0, 255]]])
        bins, scores = self.n["calibration_counts"](probs, y)
        self.assertEqual(bins[:, 0].sum().item(), 3)
        self.assertEqual(scores["brier_sum"], 0)
        self.assertAlmostEqual(self.n["calibration_report"](bins)[1], 0)

    def test_region_bootstrap_identical_regions_collapses(self):
        cm = self.n["confusion_counts"](torch.tensor([0, 1]), torch.tensor([0, 1]))
        low, high = self.n["region_bootstrap"]({"a": cm, "b": cm}, repeats=20)
        self.assertEqual((low, high), (1.0, 1.0))

    def test_occlusion_has_explicit_roi_and_does_not_mutate_input(self):
        model = self.n["UNet"](base=4).eval()
        x = torch.rand(1, 3, 32, 32)
        original = x.clone()
        roi = torch.zeros(32, 32, dtype=torch.bool)
        roi[8:16, 8:16] = True
        heatmap = self.n["occlusion_map"](model, x, class_id=1, roi=roi, grid=2)
        self.assertEqual(heatmap.shape, (32, 32))
        self.assertTrue(np.isfinite(heatmap).all())
        torch.testing.assert_close(x, original)
        with self.assertRaises(ValueError):
            self.n["occlusion_map"](model, x, class_id=1, roi=torch.zeros_like(roi), grid=2)

    def test_all_unknown_epoch_does_not_apply_weight_decay(self):
        model = self.n["UNet"](base=4)
        original = {key: value.clone() for key, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
        loader = [(torch.rand(1, 3, 32, 32), torch.full((1, 32, 32), 255), torch.tensor([0]))]
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        result = self.n["train_epoch"](model, loader, optimizer, scaler, {"amp": False}, time.perf_counter() + 10)
        self.assertTrue(result["completed_epoch"])
        self.assertEqual(result["optimizer_steps"], 0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_nll_remains_correct_for_extremely_confident_wrong_predictions(self):
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset
        class WrongModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                scores = torch.zeros(x.shape[0], 8, *x.shape[-2:])
                scores[:, 0] = 100
                return scores + self.anchor
        dataset = TensorDataset(torch.zeros(1, 3, 2, 2), torch.ones(1, 2, 2, dtype=torch.long), torch.tensor([0]))
        dataset.frame = pd.DataFrame({"region": ["a"], "image": [Path("a.tif")]})
        result = self.n["evaluate"](WrongModel(), DataLoader(dataset, batch_size=1))
        self.assertAlmostEqual(result["summary"]["nll"], 100, places=4)
        self.assertAlmostEqual(result["summary"]["nll"], result["summary"]["loss"])

    def test_reference_verification_checks_iou_and_dice(self):
        verify = self.n.get("verify_metric_reference")
        self.assertIsNotNone(verify, "No per-class IoU/Dice reference verification")
        cm = self.n["confusion_counts"](torch.tensor([0, 1, 2]), torch.tensor([0, 1, 1]))
        table, _ = self.n["metric_report"](cm)
        iou, dice = table.iou.fillna(0).to_numpy(copy=True), table.dice.fillna(0).to_numpy(copy=True)
        verify(table, iou, dice)
        wrong = dice.copy()
        wrong[1] = 1
        with self.assertRaises(AssertionError):
            verify(table, iou, wrong)
        wrong = iou.copy()
        wrong[2] = 1  # absent in ground truth but falsely predicted: must be checked
        with self.assertRaises(AssertionError):
            verify(table, wrong, dice)

    def test_final_comparison_rejects_duplicates_incomplete_and_unlisted_selection(self):
        validate = self.n.get("validate_comparison")
        self.assertIsNotNone(validate, "Final comparison gate only counts list entries")
        saved = self.n.get("PROVENANCE")
        self.n["PROVENANCE"] = {"synthetic": True}
        try:
            with tempfile.TemporaryDirectory() as folder:
                a, b = Path(folder) / "a", Path(folder) / "b"
                for directory, base in [(a, 4), (b, 8)]:
                    directory.mkdir()
                    config = {"base": base, "seed": 42}
                    (directory / "config.json").write_text(json.dumps(config))
                    (directory / "provenance.json").write_text(json.dumps(self.n["PROVENANCE"]))
                    (directory / "run_summary.json").write_text(json.dumps({"epochs_completed": 1, "best_validation_miou": 0.2}))
                    torch.save(dict(kind="experiment", config=config, provenance=self.n["PROVENANCE"],
                                    validation_miou=0.2), directory / "best.pt")
                validate([a, b], selected=a / "best.pt")
                with self.assertRaises(ValueError):
                    validate([a, a])
                with self.assertRaises(ValueError):
                    validate([a, b], selected=Path(folder) / "unlisted.pt")
                (b / "run_summary.json").write_text(json.dumps({"epochs_completed": 0, "best_validation_miou": None}))
                with self.assertRaises(ValueError):
                    validate([a, b])
        finally:
            if saved is None:
                self.n.pop("PROVENANCE", None)
            else:
                self.n["PROVENANCE"] = saved

    def test_experiment_loop_evaluation_checkpoint_and_debug_rejection(self):
        # Real optimizer/evaluation/save/load, on synthetic tensors only.
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.rand(2, 3, 32, 32)
        y = torch.randint(0, 8, (2, 32, 32))
        y[0, :2] = 255
        dataset = TensorDataset(x, y, torch.arange(2))
        dataset.frame = pd.DataFrame({"region": ["a", "b"], "image": [Path("a.tif"), Path("b.tif")]})
        loader = DataLoader(dataset, batch_size=1)
        config = dict(base=4, seed=42, lr=0.001, weight_decay=0.0001, amp=False,
                      epochs=2, max_minutes=1, patience=2, min_delta=0.0001, batch_size=1)
        overrides = dict(DEVICE=torch.device("cpu"), ENV={}, PROVENANCE={"synthetic": True},
                         parts={"train": None, "validation": None}, USE_WANDB_OFFLINE=False,
                         make_loader=lambda *args, **kwargs: loader)
        absent = object()
        saved = {key: self.n.get(key, absent) for key in overrides}
        self.n.update(overrides)
        try:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / "experiment"
                checkpoint = self.n["run_experiment"](config, directory)
                restored, info = self.n["load_checkpoint"](checkpoint)
                result = self.n["evaluate"](restored, loader, majority_class=0)
                self.assertEqual(result["summary"]["valid_pixels"], int((y != 255).sum()))
                self.assertEqual(result["summary"]["images"], 2)
                torch.testing.assert_close(sum(result["region_matrices"].values()), result["confusion"])
                curve = pd.read_csv(directory / "history.csv")
                self.assertEqual(info["validation_miou"], curve.validation_miou.max())
                self.assertTrue((directory / "best_validation_classes.csv").is_file())
                self.assertTrue(np.isfinite(result["summary"]["ece"]))
                info["kind"] = "debug_only"
                debug_path = directory / "debug.pt"
                torch.save(info, debug_path)
                with self.assertRaises(ValueError):
                    self.n["load_checkpoint"](debug_path)
        finally:
            for key, value in saved.items():
                if value is absent:
                    self.n.pop(key, None)
                else:
                    self.n[key] = value


if __name__ == "__main__":
    unittest.main()
