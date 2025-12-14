import numpy as np
import pandas as pd

class HeuristicModel:
    def __init__(self, seed):
        self.seed = seed

    @staticmethod
    def simple_tokenize(text: str) -> list[str]:
        return str(text).strip().split()
    
    @staticmethod
    def choose_span_len(n_tokens: int, rng: np.random.Generator) -> int:
        if n_tokens <= 5:
            return 1
        if n_tokens <= 15:
            return int(rng.integers(1, 3))
        if n_tokens <= 30:
            return int(rng.integers(1, 4))
        return int(rng.integers(1, 6))
    
    def random_span_prediction(self, text: str, rng: np.random.Generator) -> str:
       tokens = self.simple_tokenize(text)
       n = len(tokens)
       if n == 0:
           return ""
       k = min(self.choose_span_len(n, rng), n)
       start = int(rng.integers(0, n - k + 1))
       span = tokens[start:start + k]
       return " ".join(span)
    
    def make_predictions(self, df: pd.DataFrame) -> list[str]:
        rng = np.random.default_rng(self.seed)
        return [self.random_span_prediction(t, rng) for t in df["text"].astype(str).tolist()]