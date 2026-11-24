"""Synthetic CPU tests for the current experimental notebook; no dataset training."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/vibecode-test1.ipynb"


def load_definitions():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    namespace = {}
    setup = "".join(cells[0]["source"]).split("random.seed(SEED)")[0]
    exec(setup, namespace)
    import sys
    sys.path.insert(0, str(ROOT))
    from src.dataset import LandCoverDataset, prepare_pair
    namespace.update(device=torch.device("cpu"), LandCoverDataset=LandCoverDataset, prepare_pair=prepare_pair)
    for cell in cells:
        tree = ast.parse("".join(cell["source"]))
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), cell["id"], "exec"), namespace)
    torch.set_num_threads(2)
    return namespace


class EssentialPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = load_definitions()

    def test_odd_shape_and_all_parameter_gradients(self):
        model = self.n["ResidualASPP"](width=4)
        logits = model(torch.rand(2, 3, 33, 47))
        self.assertEqual(tuple(logits.shape), (2, 8, 33, 47))
        loss = self.n["segmentation_loss"](logits, torch.randint(0, 8, (2, 33, 47)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_unknown_pixels_do_not_change_either_loss_term(self):
        y = torch.tensor([[[0, 255], [1, 7]]])
        logits = torch.randn(1, 8, 2, 2)
        altered = logits.clone()
        altered[:, :, 0, 1] = torch.arange(8) * 100
        torch.testing.assert_close(self.n["segmentation_loss"](logits, y), self.n["segmentation_loss"](altered, y))

    def test_all_unknown_has_zero_loss_and_zero_gradient(self):
        logits = torch.randn(1, 8, 4, 4, requires_grad=True)
        loss = self.n["segmentation_loss"](logits, torch.full((1, 4, 4), 255))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_metrics_manual_absent_false_positive_and_additivity(self):
        y, p = torch.tensor([0, 0, 1, 1, 255]), torch.tensor([0, 1, 1, 2, 7])
        cm = self.n["confusion_counts"](p, y)
        scores = self.n["metric_table"](cm)
        np.testing.assert_allclose(scores.IoU[:3], [0.5, 1/3, 0])
        np.testing.assert_allclose(scores.Dice[:3], [2/3, 0.5, 0])
        self.assertTrue(scores.IoU[3:].isna().all())
        self.assertAlmostEqual(scores.IoU.mean(), 5/18)
        torch.testing.assert_close(cm, self.n["confusion_counts"](p[:2], y[:2]) + self.n["confusion_counts"](p[2:], y[2:]))

    def test_full_epoch_skips_unknown_batches_without_weight_decay(self):
        model = self.n["ResidualASPP"](4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        loader = [(torch.rand(1, 3, 32, 32), torch.full((1, 32, 32), 255))]
        self.n["train_epoch"](model, loader, optimizer)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_optimizer_updates_and_checkpoint_roundtrip(self):
        model = self.n["ResidualASPP"](4)
        x, y = torch.rand(1, 3, 32, 32), torch.randint(0, 8, (1, 32, 32))
        before = model.head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = self.n["train_epoch"](model, [(x, y)], optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.head.weight))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = self.n["ResidualASPP"](4)
            restored.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), restored(x))

    def test_evaluation_region_counts_and_existing_preprocessing(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 8, *x.shape[-2:])
                logits[:, 7] = 10
                return logits
        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for region in ["a", "b"]:
                image, mask = Path(folder) / f"{region}_image.png", Path(folder) / f"{region}_mask.png"
                raw = np.full((16, 16), 8, dtype=np.uint8)
                raw[:8, :8] = 0
                Image.fromarray(np.repeat((raw > 0)[..., None] * np.uint8(255), 3, axis=2)).save(image)
                Image.fromarray(raw).save(mask)
                rows.append(dict(region=region, image=image, mask=mask))
            frame = pd.DataFrame(rows)
            x, y = self.n["prepare_pair"](frame.iloc[0].image, frame.iloc[0]['mask'])
            expected_mask = np.repeat(np.repeat(raw, 32, axis=0), 32, axis=1)
            np.testing.assert_array_equal(y == 7, expected_mask == 8)
            # Bilinear RGB rounds corners; compare alignment on a straight edge.
            np.testing.assert_array_equal(x[0, 128] > 0.5, y[128] == 7)
            loss, cm, regions = self.n["evaluate"](Constant(), frame)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(int(cm.sum()), 2 * 512 * 512 * 3 // 4)
            self.assertAlmostEqual(self.n["metric_table"](cm).IoU.mean(), 1)
            torch.testing.assert_close(cm, sum(regions.values()))

    def test_occlusion_fixed_roi_does_not_modify_image(self):
        model = self.n["ResidualASPP"](4).eval()
        image = torch.rand(3, 32, 32)
        original = image.clone()
        roi = torch.zeros(32, 32, dtype=torch.bool)
        roi[8:16, 8:16] = True
        heatmap = self.n["occlusion"](model, image, 1, roi, grid=2)
        self.assertEqual(heatmap.shape, (32, 32))
        self.assertTrue(np.isfinite(heatmap).all())
        torch.testing.assert_close(image, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
