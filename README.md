# Reproducing BiLSTM-Attn on NASA C-MAPSS

This project reproduces the experimental pipeline described in:

> Seixas Leal et al. (2025), "Attentive Bidirectional Long Short-Term Memory Model to Predict the Remaining Useful Life of Aircraft Engines"

The goal is to rebuild the preprocessing pipeline, baseline models, and BiLSTM-Attn-style model for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan engine dataset, then compare the reproduced results with the values reported in the paper.

The default workflow starts with FD001 and can later be extended to FD002-FD004.

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your virtual environment uses a `bin` layout instead of `Scripts`, use:

```powershell
.\.venv\bin\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The NASA C-MAPSS dataset is not committed to this repository.

See the dataset setup guide:

```text
DATASET.md
```

## Quick FD001 Run

Run CNN-LSTM and BiLSTM-Attn on FD001:

```powershell
python run_experiment.py --fd FD001 --model cnn_lstm --epochs 50
python run_experiment.py --fd FD001 --model bilstm_attn --epochs 80
```

Results are written to:

```text
results/metrics.csv
results/predictions_*.csv
```

Each run is also saved under:

```text
results/runs/<run_id>/
```

Each run folder contains:

- `config.json`: experiment configuration
- `metrics.json`: evaluation metrics
- `predictions.csv`: true and predicted RUL values for each test engine
- `training_history.csv`: training and validation loss history for PyTorch models
- `checkpoint.pt`: saved PyTorch model checkpoint

If `results/metrics.csv` is open in Excel, Jupyter, or another program, it may be locked. In that case, the per-run artifacts in `results/runs/<run_id>/` are still saved normally.

## Running With Notebooks

FD001 notebook:

```text
notebooks/fd001_replication.ipynb
```

Full replication notebook for all models:

```text
notebooks/full_replication_models.ipynb
```

To run the notebooks from the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install notebook ipykernel matplotlib
jupyter notebook notebooks\full_replication_models.ipynb
```

## Summarizing Trained Results

After running experiments from the CLI or notebooks, regenerate the summary tables from `results/runs/`:

```powershell
python scripts\summarize_results.py
```

This creates or updates:

- `results/metrics_clean_updated.csv`
- `results/comparison_full_updated.csv`
- `results/comparison_fd001_updated.csv`
- `results/rmse_pivot_full_updated.csv`
- `results/leaderboard_fd001.csv`

## Running All Main Models

```powershell
python run_experiment.py --fd FD001 --model sgd
python run_experiment.py --fd FD001 --model svr
python run_experiment.py --fd FD001 --model cnn
python run_experiment.py --fd FD001 --model cnn_lstm
python run_experiment.py --fd FD001 --model bilstm
python run_experiment.py --fd FD001 --model bilstm_feature_attn
python run_experiment.py --fd FD001 --model bilstm_temporal_attn
python run_experiment.py --fd FD001 --model bilstm_attn
```

Available model names:

- `sgd`: SGD regression baseline
- `svr`: support vector regression baseline
- `cnn`: 1D CNN baseline
- `cnn_lstm`: CNN-LSTM hybrid baseline
- `bilstm`: plain BiLSTM
- `bilstm_feature_attn`: BiLSTM with feature attention
- `bilstm_temporal_attn`: BiLSTM with temporal attention
- `bilstm_attn`: dual-attention BiLSTM-inspired model

## Pipeline Design

The implemented pipeline follows the paper as closely as possible based on the available description:

- Read the 26 NASA C-MAPSS columns:
  - `unit`
  - `cycle`
  - 3 operational settings
  - 21 sensor measurements
- Select the 14 sensors used in the paper:
  - sensor 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21
- Apply z-score normalization using training-set statistics only.
- Generate sliding windows with default `--window-size 30`.
- Use the RUL value at the final cycle of each training window as the label.
- For testing, use the last available window of each test engine and the corresponding `RUL_FD00x.txt` label.
- Evaluate with:
  - RMSE
  - NASA Score
  - R2
  - rRMSE
  - KGE

Because the original paper does not provide source code, several implementation choices may affect reproducibility. These are exposed as CLI parameters, including:

- random seed
- validation split
- window size
- RUL clipping
- batch size
- number of epochs
- learning rate

## Paper RMSE Reference

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| CNN-LSTM | 16.33 | 17.33 | 17.09 | 20.25 |
| BiLSTM-Attn | 13.12 | 16.98 | 13.36 | 20.39 |

The paper reports the following average BiLSTM-Attn performance across FD001-FD004:

| Metric | Value |
|---|---:|
| RMSE | 16.0 |
| R2 | 0.82 |
| rRMSE | 20.8 |
| KGE | approximately 0.82-0.87 |
| NASA Score | 130 |

## Notes On Reproducibility

The reproduced results may differ from the paper due to:

- different random seeds
- different validation split
- unavailable Optuna tuning details
- TensorFlow implementation in the paper versus PyTorch implementation here
- possible differences in RUL clipping
- possible differences in normalization strategy
- batch size and learning rate differences
- paper code not being publicly available

Therefore, this repository should be treated as a reproducibility study and an open reconstruction of the described methodology, not an exact reproduction of the authors' private implementation.
