This folder contains experiment outputs.

Recommended lightweight files to commit:

- metrics_clean_updated.csv
- comparison_full_updated.csv
- comparison_fd001_updated.csv
- rmse_pivot_full_updated.csv
- leaderboard_fd001.csv

The older metrics.csv/comparison_*.csv files may exist for compatibility with earlier runs. If one of them is open in Excel or Jupyter, it may be locked and cannot be overwritten. The *_updated.csv files are regenerated from results/runs/*/metrics.json and exclude smoke-test runs.

Do not commit results/runs/ by default. It contains per-run artifacts such as checkpoints, predictions, and training histories.

