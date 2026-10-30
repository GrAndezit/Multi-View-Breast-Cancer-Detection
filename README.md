# Mammography Classification — Lab 4

## Environment setup

From the project root (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select the Python interpreter at `venv\Scripts\python.exe` for the notebooks. A CUDA-compatible PyTorch installation and NVIDIA GPU are recommended for training.

## Dataset paths

Download the VinDr-Mammo images (the project used a Kaggle-hosted copy) and keep the original annotation CSV files in `data/`. Prepared splits are stored in `data/splits/` as `train_pairs.csv`, `val_pairs.csv`, and `test_pairs.csv`.

The committed split CSVs contain absolute image paths under `D:\images_png`. On another machine, replace that prefix in the `CC_path` and `MLO_path` columns of **all three** split CSVs with the local PNG directory before running notebooks 02–05. The expected layout is `<PNG directory>/<study_id>/<image_id>.png`.

If you run notebook 01 to regenerate the splits, first change its `PNG_DIR` variable to your local PNG directory; the notebook will then write the corresponding paths into the CSVs. The image dataset is not included in this repository.

## Execution instructions

Open the `notebooks/` folder in VS Code and run the notebooks in order:

1. `01_data_exploration.ipynb` — inspect labels and prepare splits.
2. `02_single-view_models.ipynb` — train single-view baselines.
3. `03_multi-view.ipynb` — train and compare fusion models.
4. `04_ablation.ipynb` — run loss, resolution, and augmentation experiments.
5. `05_evaluation.ipynb` — evaluate saved checkpoints and generate figures.

Run from the project environment with the project folders in their documented locations. Training requires the dataset; evaluation additionally requires saved checkpoints in `models/` (including `models/final/`). Outputs are stored in `results/`.

The selected false-positive, false-negative, and high-confidence cases are reviewed in `results/evaluation/qualitative_error_review.csv`. To regenerate any missing Grad-CAM figures for these cases after placing the seed-42 multi-view checkpoint in `models/final/`, run `python scripts/generate_error_gradcam.py` from the project root.
