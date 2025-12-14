import subprocess
import sys

import fire


def dvc_pull(enabled: bool):
    if not enabled:
        print("Skipping dvc pull")
        return
    print("Running: dvc pull")
    subprocess.run(["dvc", "pull"], check=True)


class Commands:
    def train(self, *hydra_args, dvc: bool = False):
        dvc_pull(dvc)
        self._run("tweet_sentiment_extraction.train", hydra_args)

    def infer(self, *hydra_args, dvc: bool = False):
        dvc_pull(dvc)
        self._run("tweet_sentiment_extraction.infer", hydra_args)

    def dummy_train(self, *hydra_args, dvc: bool = False):
        dvc_pull(dvc)
        self._run("tweet_sentiment_extraction.dummy_train", hydra_args)

    def _run(self, module: str, hydra_args):
        cmd = [sys.executable, "-m", module, *hydra_args]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    fire.Fire(Commands)
