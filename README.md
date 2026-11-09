# Land Cover Mapping Using Semantic Segmentation

Fontys ICT Semester 3 — Main Challenge.

**Research question:** How accurately can CNN-based semantic segmentation models classify land-cover types, and how well do they generalize to unseen geographical regions?

## Project folders

| Folder / file | Purpose |
| --- | --- |
| `data/OpenEarthMap_wo_xBD/` | Original extracted dataset. Preserve these files unchanged. |
| `docs/` | Project proposal, domain research and future reports. |
| `notebooks/` | Future exploratory analysis and experiments. Currently empty. |
| `src/` | Future reusable Python code. Currently empty. |
| `tests/` | Future checks for data processing and metrics. Currently empty. |
| `results/figures/` | Figures produced by analysis and evaluation. |
| `results/tables/` | Inventories, class counts and measured results. |
| `checkpoints/` | Future saved model weights. |
| `app/` | Future demonstration application, after model validation. |
| `AGENTS.md` | Collaboration guidelines. |
| `CODEX_PROJECT_BRIEF.md` | Full project context and research requirements. |

## Current status

Only the project folders and documentation have been prepared. No analysis notebook, training code or application has been created as part of this setup. No models have been trained.

The downloaded `OpenEarthMap.zip` matches the [official archive](https://zenodo.org/records/7223446), with MD5 `64155d1dc9d3b68536063f79878e1a67`.

The archive contains 3,838 RGB images and 3,500 masks across 97 region folders, yielding 2,687 image/mask pairs (2,303 official training pairs and 384 validation pairs). The 1,162 missing RGB files match the xBD list; the 1,500 absent masks match the official test IDs. The original download is complete, but it does not contain 5,000 usable labeled pairs.

The original ZIP remains in Downloads; this project holds the extracted dataset. No additional dataset downloads have been made. Data licenses vary by source; consult the official dataset record before redistributing samples.

## Next step

Agree on the first small exploration step before writing code. Start with dataset paths and inventory, followed by a few image/mask examples. Full file decoding, label checks, content duplicates, class distributions and geographical split design remain to be investigated.

An isolated project virtual environment has been created at `.venv/` using Python 3.14.5. Its interpreter is `.venv/Scripts/python.exe`; select it in VS Code. Only pip is installed in this environment. EDA dependencies are pending installation, and training-library compatibility has not been validated.
