from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "Dataset NASA CMAPPS"
    / "6.+Turbofan+Engine+Degradation+Simulation+Data+Set"
    / "6. Turbofan Engine Degradation Simulation Data Set"
    / "CMAPSSData"
)

COLUMN_NAMES = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

SELECTED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

PAPER_RMSE = {
    "cnn_lstm": {"FD001": 16.33, "FD002": 17.33, "FD003": 17.09, "FD004": 20.25},
    "bilstm_attn": {"FD001": 13.12, "FD002": 16.98, "FD003": 13.36, "FD004": 20.39},
}
