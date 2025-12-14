import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from tweet_sentiment_extraction.modules.heuristic_model import HeuristicModel
from tweet_sentiment_extraction.modules.metrics import evaluate_jaccard


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    train_csv_path = to_absolute_path(cfg.data_loading.train_csv)
    df = pd.read_csv(train_csv_path)

    required_cols = {"text", "sentiment", "selected_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["text"] = df["text"].astype(str)
    df["selected_text"] = df["selected_text"].astype(str)

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=cfg.training.seed,
        stratify=df["sentiment"],
    )

    baseline = HeuristicModel(seed=cfg.training.seed)
    val_preds = baseline.make_predictions(val_df)

    val_jacc = evaluate_jaccard(val_df, val_preds)
    print(f"Baseline Jaccard score: {val_jacc:.6f}")


if __name__ == "__main__":
    main()
