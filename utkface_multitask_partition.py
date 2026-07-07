import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Initialize UTKFace multitask SISA with data shards and unlearning scenarios."
    )
    parser.add_argument("--container", required=True, help="Container name")
    parser.add_argument(
        "--dataset",
        default="datasets/UTKFace/datasetfile_ver2",
        help="Path to UTKFace datasetfile",
    )
    parser.add_argument("--shards", type=int, default=3, help="Number of data shards")
    parser.add_argument(
        "--scenarios",
        default="0,100,500",
        help="Comma-separated unlearning scenario sizes",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    container_dir = f"containers/{args.container}"
    os.makedirs(f"{container_dir}/cache", exist_ok=True)
    os.makedirs(f"{container_dir}/times", exist_ok=True)
    os.makedirs(f"{container_dir}/outputs", exist_ok=True)

    with open(f"{container_dir}/times/null.time", "w") as f:
        f.write("0\n")

    subprocess.run(
        [
            sys.executable,
            "distribution_safe.py",
            "--shards",
            str(args.shards),
            "--distribution",
            "uniform",
            "--container",
            args.container,
            "--dataset",
            args.dataset,
            "--label",
            "0",
        ],
        check=True,
    )

    scenarios = [int(x.strip()) for x in args.scenarios.split(",") if x.strip()]
    for n_requests in scenarios:
        subprocess.run(
            [
                sys.executable,
                "distribution_safe.py",
                "--requests",
                str(n_requests),
                "--distribution",
                "uniform",
                "--container",
                args.container,
                "--dataset",
                args.dataset,
                "--label",
                str(n_requests),
            ],
            check=True,
        )
        print(f"Created requestfile:{n_requests}.npy")

    with open(args.dataset, "r") as f:
        datasetfile = json.loads(f.read())

    meta = {
        "mode": "joint_multitask",
        "tasks": ["gender", "age", "race"],
        "num_classes": {"gender": 2, "age": 3, "race": 5},
        "shards": args.shards,
        "scenarios": scenarios,
        "nb_train": datasetfile["nb_train"],
        "description": (
            "Each data shard trains one MultiTaskModel predicting gender+age+race jointly. "
            "Unlearning scenarios forget N random training samples distributed across shards."
        ),
    }

    with open(f"{container_dir}/multitask_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Created UTKFace joint-multitask partition: {args.shards} data shards")
    print(f"Unlearning scenarios: {scenarios}")


if __name__ == "__main__":
    main()
