# Land Cover Mapping Using Semantic Segmentation

Fontys ICT Semester 3 — Main Challenge.

**Research question:** How accurately can CNN-based semantic segmentation models classify land-cover types, and how well do they generalize to unseen geographical regions?

## Project Overview

This project investigates how deep learning can identify land-cover types in aerial and satellite images. Using the OpenEarthMap dataset, it focuses on semantic segmentation: assigning a class, such as building, road, tree or water, to each pixel in an image.

## Problem

Manually mapping land cover is time-consuming. Automatic classification is also challenging because different surfaces can look similar, while landscapes, lighting and image resolution vary between locations. A model that performs well on familiar regions may struggle in a new geographical area.

## Proposed Solution

The project aims to train and compare CNN-based segmentation models, starting with U-Net, and evaluate their predictions using overall and per-class metrics. Separate geographical regions will be used for training, validation and testing to assess performance on unseen locations.

A simple application is planned to let users upload an aerial image and view its predicted land-cover mask and class percentages.

## Why It Matters

Land-cover information can support environmental research and urban planning, for example by helping identify the distribution of vegetation and built surfaces. This project explores the potential and limitations of automating that analysis, with particular attention to reliability across geographical regions. Its predictions are estimates from an academic prototype, rather than validated planning evidence.

## Continue on another Windows PC

Install Git, Python and VS Code with the Python and Jupyter extensions. The source
PC uses Python 3.14.5 and a CPU build of PyTorch; its package versions are recorded
in [docs/environment-local.txt](docs/environment-local.txt). This is an environment
record, not a portable lockfile. `requirements.txt` lists the direct dependencies;
a fresh installation may resolve different versions and must pass the checks below.

Clone the project and create a new environment from PowerShell:

```powershell
git clone https://github.com/jarimbafrancisco-arch/Land-Cover-Segmentation-Mapping.git
cd Land-Cover-Segmentation-Mapping
py -3.14 -m venv .venv
```

Choose the PyTorch build before installing the remaining packages. For a CPU-only
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
to choose Windows, Pip and the appropriate CUDA build. Run its command using
`.\.venv\Scripts\python.exe -m pip` so it installs into this environment. The
current notebooks require `torch`; `torchvision` and `torchaudio` are not used.
Then install the remaining dependencies and check the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Select `.venv\Scripts\python.exe` as the notebook kernel in VS Code. Do not copy
the old `.venv`: recreate it on the destination PC. The synthetic tests do not
require OpenEarthMap and do not start a dataset training run.

### Files to transfer separately

Git contains the notebooks, reusable dataset code and the saved split manifest.
The following local files are deliberately excluded and must be copied separately
if needed:

- `data/OpenEarthMap_wo_xBD/`: original dataset, keeping its region/image/label folders.
- `results/models/`, `checkpoints/` and other `.pt`/`.pth`/`.ckpt` files: existing weights.
- `results/vibecode-test1/`: local experiment evidence and historical backups.
- `AGENTS.md` and `CODEX_PROJECT_BRIEF.md`: local project instructions for Codex.

You can copy the extracted dataset directly, or transfer the existing
`data/kaggle_baseline.zip` and extract its `OpenEarthMap_wo_xBD/` folder under
`data/`. There is no need to transfer both copies. Keep the versioned
`results/tables/split_manifest.csv`; do not regenerate the geographical splits
just because the computer changed.

Expected layout:

```text
Land-Cover-Segmentation-Mapping/
  data/OpenEarthMap_wo_xBD/<region>/images/*.tif
  data/OpenEarthMap_wo_xBD/<region>/labels/*.tif
  notebooks/03_baseline.ipynb
  src/dataset.py
  results/tables/split_manifest.csv
```

### First notebook checks

The numbered notebooks use paths relative to `notebooks/`. Check `Path.cwd()`
before running them; if the kernel starts in the repository root, run `%cd notebooks`
once. The saved manifest uses Windows paths; Linux/Kaggle requires the path
conversion discussed separately, while the same relative layout works on Windows.

Open `03_baseline.ipynb`, restart its kernel and run only through **Optimizer and
Backpropagation**. Check the batch shapes, finite logits of shape `(1, 8, 512, 512)`
and finite loss. Saved outputs are historical evidence, not a fresh environment
check. Do not use **Run All**: subsequent cells train the model, save a checkpoint
and create a dataset ZIP. The baseline training loop currently reports training
metrics only; validation and the remaining scientific checks are still required
before full experiments. Keep test regions held out.

`vibecode-test1.ipynb` is a separate experimental demonstration. Its current tests
are in `tests/test_vibecode_test1.py`; tests for its retired implementation are
preserved in `tests/archive/` and are not part of test discovery. Running
`tests/run_vibecode_notebook.py` executes that whole experimental notebook,
including its training cells; it is not an environment smoke test.
