# Reproducing BiLSTM-Attn on NASA C-MAPSS

Project nay dung lai pipeline trong paper:

> Seixas Leal et al. (2025), "Attentive Bidirectional Long Short-Term Memory Model to Predict the Remaining Useful Life of Aircraft Engines"

Muc tieu la chay replication tren NASA C-MAPSS va so sanh ket qua voi paper. Code mac dinh chay FD001 truoc, sau do co the mo rong FD002-FD004.

## Cai dat

Khuyen dung Python bundled/Windows venv co thu muc `Scripts`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu venv cua ban tao layout `bin` thay vi `Scripts`, hay dung:

```powershell
.\.venv\bin\Activate.ps1
pip install -r requirements.txt
```

Trong workspace hien tai, Codex da tao san `.venv-codex` bang Python bundled:

```powershell
.\.venv-codex\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Dataset NASA C-MAPSS khong duoc commit vao Git. Xem huong dan tai:

```text
DATASET.md
```

## Chay nhanh FD001

```powershell
python run_experiment.py --fd FD001 --model cnn_lstm --epochs 50
python run_experiment.py --fd FD001 --model bilstm_attn --epochs 80
```

Ket qua duoc ghi vao `results/metrics.csv` va prediction tung engine vao `results/predictions_*.csv`.
Moi run cung duoc luu rieng trong `results/runs/<run_id>/`, gom:

- `config.json`
- `metrics.json`
- `predictions.csv`
- `training_history.csv` cho PyTorch models
- `checkpoint.pt` cho PyTorch models

Neu `results/metrics.csv` dang mo trong Excel/Jupyter va bi lock, artifact rieng trong `results/runs/<run_id>/` van duoc luu binh thuong.

## Chay bang notebook

Notebook FD001:

```text
notebooks/fd001_replication.ipynb
```

Notebook full replication tat ca model:

```text
notebooks/full_replication_models.ipynb
```

Neu muon chay notebook bang venv cua project:

```powershell
.\.venv-codex\Scripts\Activate.ps1
pip install notebook ipykernel matplotlib
jupyter notebook notebooks\fd001_replication.ipynb
```

## Tong hop ket qua da train

Sau khi notebook/CLI tao cac run trong `results/runs/`, co the tao lai cac bang summary bang:

```powershell
python scripts\summarize_results.py
```

Script nay tao/cap nhat:

- `results/metrics_clean_updated.csv`
- `results/comparison_full_updated.csv`
- `results/comparison_fd001_updated.csv`
- `results/rmse_pivot_full_updated.csv`
- `results/leaderboard_fd001.csv`

## Chay tat ca model chinh

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

## Thiet ke pipeline

- Doc 26 cot NASA C-MAPSS: `unit`, `cycle`, 3 operational settings, 21 sensors.
- Chon 14 sensor theo paper: sensor 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21.
- Chuan hoa z-score bang thong ke train only.
- Tao sliding window, mac dinh `--window-size 30`.
- Train label la RUL tai cycle cuoi cua window.
- Test dung window cuoi cung cua moi unit va `RUL_FD00x.txt`.
- Metrics: RMSE, NASA Score, R2, rRMSE, KGE.

Paper khong public code, nen cac diem de lam lech ket qua duoc de thanh tham so CLI: seed, validation split, window size, RUL clipping, batch size, epochs, learning rate.

## RMSE paper de doi chieu

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| CNN-LSTM | 16.33 | 17.33 | 17.09 | 20.25 |
| BiLSTM-Attn | 13.12 | 16.98 | 13.36 | 20.39 |

Trung binh BiLSTM-Attn paper bao cao: RMSE 16.0, R2 0.82, rRMSE 20.8, KGE khoang 0.82-0.87, NASA Score 130.
