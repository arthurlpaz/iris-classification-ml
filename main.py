import argparse
import subprocess


def run_train():
    subprocess.run(["python", "train.py"])


def run_api():
    subprocess.run(["uvicorn", "api.app:app", "--reload"])


def run_all():
    run_train()
    run_api()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "api", "all"],
        required=True,
        help="Choose what to run"
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "api":
        run_api()
    elif args.mode == "all":
        run_all()