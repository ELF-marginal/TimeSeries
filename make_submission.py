import argparse
import os
import zipfile


PRED_FILES = ["pred_96.npy", "pred_192.npy", "pred_336.npy", "pred_720.npy"]


def add_file(zipf, path, arcname):
    if os.path.exists(path):
        zipf.write(path, arcname)


def main():
    parser = argparse.ArgumentParser(description="Package code, notes, checkpoints and prediction files.")
    parser.add_argument("--name", required=True, help="Zip file name without .zip")
    parser.add_argument("--pred_dir", default="predictions")
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_dir = os.path.join(base_dir, args.pred_dir)
    zip_path = os.path.join(base_dir, args.output_dir, args.name + ".zip")

    missing = [name for name in PRED_FILES if not os.path.exists(os.path.join(pred_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing prediction files in {pred_dir}: {missing}")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for name in PRED_FILES:
            add_file(zipf, os.path.join(pred_dir, name), name)

        for root, _, files in os.walk(base_dir):
            skip_dirs = {"train", "test_demo", "predictions", "__pycache__"}
            rel_root = os.path.relpath(root, base_dir)
            if any(part in skip_dirs for part in rel_root.split(os.sep)):
                continue
            for file_name in files:
                if file_name.endswith((".pyc", ".zip")):
                    continue
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, base_dir)
                add_file(zipf, full_path, rel_path)

    print(f"Saved submission package: {zip_path}")


if __name__ == "__main__":
    main()
