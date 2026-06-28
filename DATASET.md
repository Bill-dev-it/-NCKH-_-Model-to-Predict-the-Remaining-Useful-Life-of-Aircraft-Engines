# Dataset Setup

This project expects the NASA C-MAPSS text files to be available locally.

The default data directory used by the code is:

```text
Dataset NASA CMAPPS/
  6.+Turbofan+Engine+Degradation+Simulation+Data+Set/
    6. Turbofan Engine Degradation Simulation Data Set/
      CMAPSSData/
        train_FD001.txt
        test_FD001.txt
        RUL_FD001.txt
        train_FD002.txt
        test_FD002.txt
        RUL_FD002.txt
        train_FD003.txt
        test_FD003.txt
        RUL_FD003.txt
        train_FD004.txt
        test_FD004.txt
        RUL_FD004.txt
```

The dataset folder is intentionally excluded from Git because it is external data and can make the repository unnecessarily large.

If your dataset is stored somewhere else, pass the path explicitly:

```powershell
python run_experiment.py --fd FD001 --model cnn_lstm --data-dir "C:\path\to\CMAPSSData"
```

