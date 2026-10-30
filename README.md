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

The `CC_path` and `MLO_path` columns must point to existing image files on your machine. If the image directory changes, update these paths before running the notebooks. The image dataset is not included in this repository.

## Execution instructions

Open the `notebooks/` folder in VS Code and run the notebooks in order:

1. `01_data_exploration.ipynb` — inspect labels and prepare splits.
2. `02_single-view_models.ipynb` — train single-view baselines.
3. `03_multi-view.ipynb` — train and compare fusion models.
4. `04_ablation.ipynb` — run loss, resolution, and augmentation experiments.
5. `05_evaluation.ipynb` — evaluate saved checkpoints and generate figures.

Run from the project environment with the project folders in their documented locations. Training requires the dataset; evaluation additionally requires saved checkpoints in `models/` (including `models/final/`). Outputs are stored in `results/`.
