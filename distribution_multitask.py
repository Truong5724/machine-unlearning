import argparse
import json
import os
import sys

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "--shards",
    default=None,
    type=int,
    help="Split the dataset into shards and create splitfile",
)
parser.add_argument(
    "--requests",
    default=None,
    type=int,
    help="Generate unlearning requests and apply them to splitfile",
)
parser.add_argument(
    "--distribution",
    default="uniform",
    help="Distribution for shards/requests, default uniform",
)
parser.add_argument("--algo", help='PLS-GAP algorithm parameter, format: "<key>:<value>"')
parser.add_argument("--container", default="default", help="Name of the container")
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile",
)
parser.add_argument("--label", default="latest", help="Label, default latest")
parser.add_argument(
    "--unlearn",
    nargs="+",
    default=None,
    help="Examples: gender:0 age:2 race:1,4",
)
args = parser.parse_args()


with open(args.dataset) as f:
    datasetfile = json.loads(f.read())


def ensure_object_array(items):
    return np.array([np.asarray(item, dtype=np.int64) for item in items], dtype=object)


if args.shards is not None:
    if args.distribution == "uniform":
        partition = np.split(
            np.arange(0, datasetfile["nb_train"]),
            [t * (datasetfile["nb_train"] // args.shards) for t in range(1, args.shards)],
        )
        partition = ensure_object_array(partition)
        np.save(f"containers/{args.container}/splitfile.npy", partition)
        requests = ensure_object_array([np.array([], dtype=np.int64) for _ in range(args.shards)])
        np.save(f"containers/{args.container}/requestfile:{args.label}.npy", requests)
        print(f"✅ Created {args.shards} uniform shards")
    else:
        # Phần non-uniform (PLS-GAP) giữ nguyên logic phức tạp của bạn
        def mass(index):
            if args.distribution.split(":")[0] == "exponential":
                lbd = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else -np.log(0.05) / datasetfile["nb_train"]
                )
                return np.exp(-lbd * index) - np.exp(-lbd * (index + 1))
            if args.distribution.split(":")[0] == "pareto":
                a = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else 1.16
                )
                return a / ((index + 1) ** (a + 1))
            return 1.0

        weights = mass(np.arange(0, datasetfile["nb_train"]))
        indices = np.argsort(weights)
        queue = np.array([weights[indices], np.ones(weights.shape)]).transpose()
        partition = [np.array([index]) for index in indices]

        bottom_queue = queue.shape[0]
        lim = (
            int(float(args.algo.split(":")[1]) * datasetfile["nb_train"])
            if args.algo and len(args.algo.split(":")) > 1
            else int(0.01 * datasetfile["nb_train"])
        )

        for _ in range(datasetfile["nb_train"] - args.shards):
            w1 = queue[0]
            w2 = queue[1]
            l1 = partition[0]
            l2 = partition[1]

            partition = partition[2:]
            queue = queue[2:]
            bottom_queue -= 2

            merged_weight = w1 + w2

            if merged_weight[1] < lim:
                offset_array = np.where(queue[:bottom_queue, 1] >= merged_weight[1])
                limit_array = np.where(queue[:bottom_queue, 1] > merged_weight[1])
                offset = offset_array[0][0] if offset_array[0].shape[0] > 0 else bottom_queue
                limit = limit_array[0][0] if limit_array[0].shape[0] > 0 else bottom_queue
                position_array = np.where(queue[offset:limit][:, 0] >= merged_weight[0])
                position = position_array[0][0] if position_array[0].shape[0] > 0 else bottom_queue
                bottom_queue += 1
            else:
                position_array = np.where(queue[bottom_queue:][:, 0] >= merged_weight[0])
                position = position_array[0][0] if position_array[0].shape[0] > 0 else queue.shape[0]

            queue = np.insert(queue, position, merged_weight, axis=0)
            partition = partition[:position] + [np.concatenate((l1, l2))] + partition[position:]

        partition = ensure_object_array(partition)
        np.save(f"containers/{args.container}/splitfile.npy", partition)
        requests = ensure_object_array([np.array([], dtype=np.int64) for _ in range(len(partition))])
        np.save(f"containers/{args.container}/requestfile:{args.label}.npy", requests)


if args.requests is not None:
    if args.distribution == "reset":
        partition = np.load(f"containers/{args.container}/splitfile.npy", allow_pickle=True)
        requests = ensure_object_array([np.array([], dtype=np.int64) for _ in range(len(partition))])
        np.save(f"containers/{args.container}/requestfile:{args.label}.npy", requests)
    else:
        partition = np.load(f"containers/{args.container}/splitfile.npy", allow_pickle=True)

        if args.unlearn is not None:

            dataset_dir = os.path.dirname(args.dataset)
            sys.path.insert(0, dataset_dir)

            dataloader_module = __import__(datasetfile["dataloader"])

            all_indices = np.arange(datasetfile["nb_train"])

            _, labels = dataloader_module.load_multitask(
                all_indices,
                category="train",
            )

            all_requests = np.array([], dtype=np.int64)

            for rule in args.unlearn:

                task, cls_string = rule.split(":")

                classes = [
                    int(x)
                    for x in cls_string.split(",")
                ]

                y_train = labels[task]

                for cls in classes:

                    idx = np.where(y_train == cls)[0]

                    all_requests = np.concatenate(
                        (
                            all_requests,
                            idx,
                        )
                    )

                    print(
                        f"{task} class {cls}: {len(idx)} samples"
                    )

            all_requests = np.unique(all_requests)

            print(f"\nTotal forgot samples: {len(all_requests)}")
        else:
            np.random.seed(1)
            if args.distribution.split(":")[0] == "exponential":
                lbd = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else -np.log(0.05) / datasetfile["nb_train"]
                )
                all_requests = np.random.exponential(1 / lbd, args.requests).astype(int)
            elif args.distribution.split(":")[0] == "pareto":
                a = (
                    float(args.distribution.split(":")[1])
                    if len(args.distribution.split(":")) > 1
                    else 1.16
                )
                all_requests = np.random.pareto(a, args.requests).astype(int)
            else:
                all_requests = np.random.randint(0, datasetfile["nb_train"], args.requests)

        requests = []
        for shard in range(len(partition)):
            requests.append(np.intersect1d(partition[shard], all_requests))

        np.save(
            f"containers/{args.container}/requestfile:{args.label}.npy",
            np.array(requests, dtype=object),
        )
        print(
            f"✅ Created requestfile:{args.label} "
            f"with {len(all_requests)} unlearning samples"
        )

