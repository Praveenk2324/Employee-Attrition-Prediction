import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def ingest(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:

    raw_path = config['paths']['raw_data']
    train_out = config['paths']['train_data']
    test_out = config['paths']['test_data']

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    drop_cols = [cols for cols in config['data']['drop_columns'] if cols in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info(f"Dropped columns: {drop_cols}")

    target_col = config['data']['target_column']
    df[target_col] = (df[target_col] == "Yes").astype(int)
    logger.info(f"Class distribution :\n{df[target_col].value_counts()}")

    train_df, test_df = train_test_split(df, test_size=config['data']['test_size'], random_state=config['data']['random_state'], stratify=df[target_col])

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"Train: {train_df.shape} → {train_out}")
    logger.info(f"Test:  {test_df.shape}  → {test_out}")

    return train_df, test_df

if __name__ == "__main__":
    cfg = load_config()
    ingest(cfg)