from pathlib import Path
import shutil

import kagglehub


BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset" / "source"

DATASETS = {
    "wcukierski/enron-email-dataset": DATASET_ROOT / "enron",
    "naserabdullahalam/phishing-email-dataset": DATASET_ROOT / "phishing",
}


def copy_downloaded_dataset(source_path: str, target_dir: Path) -> None:
    source_dir = Path(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def main() -> None:
    for dataset_name, target_dir in DATASETS.items():
        downloaded_path = kagglehub.dataset_download(dataset_name)
        copy_downloaded_dataset(downloaded_path, target_dir)

        print(f"Downloaded: {dataset_name}")
        print(f"Kaggle cache: {downloaded_path}")
        print(f"Project path: {target_dir}")


if __name__ == "__main__":
    main()
