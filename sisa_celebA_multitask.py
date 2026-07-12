import argparse
import importlib
import json
import os
from glob import glob
from time import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from tqdm import tqdm


from architectures.celebA_multitask import CelebAMultiTaskModel
from aggregation_multitask_celebA import binary_metrics
from sharded import getShardHash, sizeOfShard


NUM_ATTRIBUTES = 27
TASKS = [
    f"attr_{i}"
    for i in range(NUM_ATTRIBUTES)
]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset_config(datasetfile_path):

    with open(datasetfile_path,"r") as f:
        datasetfile=json.load(f)


    module_name=".".join(
        datasetfile_path.replace("\\","/")
        .split("/")[:-1]
        +
        [
            datasetfile["dataloader"]
        ]
    )


    dataloader=importlib.import_module(
        module_name
    )


    return datasetfile,dataloader



# ==========================================================
# FETCH TRAIN BATCH
# ==========================================================

def fetch_celeba_batch(
        container,
        label,
        shard,
        batch_size,
        dataset,
        offset=0,
        until=None
):


    shards=np.load(
        f"containers/{container}/splitfile.npy",
        allow_pickle=True
    )


    requests=np.load(
        f"containers/{container}/requestfile:{label}.npy",
        allow_pickle=True
    )


    _,dataloader=load_dataset_config(dataset)


    if until is None or until > len(shards[shard]):
        until=len(shards[shard])


    limit=offset


    while limit <= until-batch_size:

        idx_range=shards[shard][
            limit:
            limit+batch_size
        ]

        limit+=batch_size


        indices=np.setdiff1d(
            idx_range,
            requests[shard]
        )


        if len(indices)>0:

            yield dataloader.load(
                indices,
                category="train"
            )



    if limit < until:

        indices=np.setdiff1d(
            shards[shard][limit:until],
            requests[shard]
        )


        if len(indices)>0:

            yield dataloader.load(
                indices,
                category="train"
            )



# ==========================================================
# OPTIMIZER
# ==========================================================

def make_optimizer(
        model,
        name,
        lr
):

    if name=="adam":

        return Adam(
            model.parameters(),
            lr=lr
        )


    elif name=="sgd":

        return SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )


    raise ValueError(
        "optimizer not supported"
    )



# ==========================================================
# LOSS
# ==========================================================

def multitask_loss(
        outputs,
        labels,
        loss_fns
):

    loss=0


    for i in range(NUM_ATTRIBUTES):

        logits=outputs[
            TASKS[i]
        ]


        y=torch.from_numpy(
            labels[:,i]
        ).long().to(
            logits.device
        )


        loss += loss_fns[i](
            logits,
            y
        )


    return loss
# ==========================================================
# TRAIN
# ==========================================================

def train(args):

    datasetfile, dataloader = load_dataset_config(
        args.dataset
    )


    input_shape = tuple(
        datasetfile["input_shape"]
    )


    device=torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


    model=CelebAMultiTaskModel(
        input_shape=input_shape,
        dropout_rate=args.dropout_rate,
        num_attributes=NUM_ATTRIBUTES
    ).to(device)



    shard_size=sizeOfShard(
        args.container,
        args.shard
    )


    if shard_size==0:
        print(
            f"Shard {args.shard} empty"
        )
        return



    slice_size=max(
        1,
        shard_size//args.slices
    )


    avg_epochs_per_slice = (
        2*args.slices
        /
        (args.slices+1)
        *
        args.epochs
        /
        args.slices
    )



    loss_fns=[
        CrossEntropyLoss()
        for _ in range(NUM_ATTRIBUTES)
    ]


    optimizer=make_optimizer(
        model,
        args.optimizer,
        args.learning_rate
    )


    loaded=False
    elapsed_time=0.0



    for sl in tqdm(
        range(args.slices),
        desc=f"Shard {args.shard}"
    ):



        slice_hash=getShardHash(
            args.container,
            args.label,
            args.shard,
            until=(sl+1)*slice_size
        )



        final_ckpt=(
            f"containers/{args.container}/cache/"
            f"{slice_hash}.pt"
        )


        final_time=(
            f"containers/{args.container}/times/"
            f"{slice_hash}.time"
        )



        if os.path.exists(final_ckpt):

            if sl==args.slices-1:

                link=(
                    f"containers/{args.container}/cache/"
                    f"shard-{args.shard}:{args.label}.pt"
                )


                if os.path.exists(link) or os.path.islink(link):
                    os.remove(link)


                os.symlink(
                    f"{slice_hash}.pt",
                    link
                )


            continue




        start_epoch=0



        slice_epochs=(
            int((sl+1)*avg_epochs_per_slice)
            -
            int(sl*avg_epochs_per_slice)
        )



        if not loaded:


            recovery=glob(
                f"containers/{args.container}/cache/"
                f"{slice_hash}_*.pt"
            )



            if recovery:

                model.load_state_dict(
                    torch.load(
                        recovery[0],
                        map_location=device
                    )
                )


                start_epoch=int(
                    recovery[0]
                    .split("_")[-1]
                    .split(".")[0]
                )



            elif sl>0:


                prev_hash=getShardHash(
                    args.container,
                    args.label,
                    args.shard,
                    until=sl*slice_size
                )


                prev_ckpt=(
                    f"containers/{args.container}/cache/"
                    f"{prev_hash}.pt"
                )


                if os.path.exists(prev_ckpt):

                    model.load_state_dict(
                        torch.load(
                            prev_ckpt,
                            map_location=device
                        )
                    )



            loaded=True





        until=(
            (sl+1)*slice_size
            if sl < args.slices-1
            else None
        )



        train_time=0.0



        for epoch in tqdm(
            range(
                start_epoch,
                slice_epochs
            ),
            leave=False
        ):


            model.train()


            running_loss=0.0



            for images, labels in fetch_celeba_batch(
                args.container,
                args.label,
                args.shard,
                args.batch_size,
                args.dataset,
                until=until
            ):

                x = torch.from_numpy(
                    images
                ).float().to(device)


    # ==============================
    # Measure ONLY forward + backward
    # ==============================

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                fb_start = time()


                outputs = model(x)


                loss = multitask_loss(
                    outputs,
                    labels,
                    loss_fns
                )


                optimizer.zero_grad()

                loss.backward()


                if torch.cuda.is_available():
                    torch.cuda.synchronize()


                train_time += (
                    time() - fb_start
                )


                # optimizer update không tính thời gian
                optimizer.step()


                running_loss += loss.item()




            print(
                f"[Shard {args.shard}] "
                f"[Slice {sl}] "
                f"Epoch {epoch+1} "
                f"loss={running_loss:.4f} "
                f"time={train_time:.2f}s"
            )




            if (
                args.chkpt_interval!=-1
                and
                epoch % args.chkpt_interval
                ==
                args.chkpt_interval-1
            ):


                torch.save(
                    model.state_dict(),
                    f"containers/{args.container}/cache/"
                    f"{slice_hash}_{epoch}.pt"
                )





        torch.save(
            model.state_dict(),
            final_ckpt
        )


        with open(final_time,"w") as f:

            f.write(
                str(train_time+elapsed_time)
            )




        if sl==args.slices-1:


            link=(
                f"containers/{args.container}/cache/"
                f"shard-{args.shard}:{args.label}.pt"
            )


            if os.path.exists(link) or os.path.islink(link):
                os.remove(link)


            os.symlink(
                f"{slice_hash}.pt",
                link
            )



            time_link=(
                f"containers/{args.container}/times/"
                f"shard-{args.shard}:{args.label}.time"
            )


            if os.path.exists(time_link) or os.path.islink(time_link):
                os.remove(time_link)


            os.symlink(
                f"{slice_hash}.time",
                time_link
            )
# ==========================================================
# TEST / PREDICT
# ==========================================================

@torch.no_grad()
def test(args):

    datasetfile, dataloader = load_dataset_config(
        args.dataset
    )


    input_shape = tuple(
        datasetfile["input_shape"]
    )


    device=torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


    model=CelebAMultiTaskModel(
        input_shape=input_shape,
        dropout_rate=args.dropout_rate,
        num_attributes=NUM_ATTRIBUTES
    ).to(device)



    ckpt=(
        f"containers/{args.container}/cache/"
        f"shard-{args.shard}:{args.label}.pt"
    )


    if not os.path.exists(ckpt):

        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt}"
        )



    model.load_state_dict(
        torch.load(
            ckpt,
            map_location=device
        )
    )


    model.eval()



    test_indices=np.arange(
        datasetfile["nb_test"],
        dtype=np.int64
    )


    _,test_labels=dataloader.load(
        test_indices,
        category="test"
    )



    scores=np.zeros(
        (
            len(test_indices),
            NUM_ATTRIBUTES
        ),
        dtype=np.float32
    )



    for start in range(
        0,
        len(test_indices),
        args.batch_size
    ):


        batch_ids=test_indices[
            start:
            start+args.batch_size
        ]



        images,_=dataloader.load(
            batch_ids,
            category="test"
        )



        x=torch.from_numpy(
            images
        ).float().to(device)



        outputs=model(x)



        batch_scores=[]


        for i in range(NUM_ATTRIBUTES):


            logits=outputs[
                TASKS[i]
            ]



            # lấy xác suất class 1
            prob=torch.softmax(
                logits,
                dim=1
            )[:,1]



            batch_scores.append(
                prob.cpu().numpy()
            )



        batch_scores=np.stack(
            batch_scores,
            axis=1
        )



        scores[
            start:start+len(batch_ids)
        ]=batch_scores





    output_path=(
        f"containers/{args.container}/outputs/"
        f"shard-{args.shard}:{args.label}.npy"
    )



    np.save(
        output_path,
        scores
    )



    print("="*70)
    print(
        f"Shard {args.shard} prediction saved"
    )
    print(
        "Output shape:",
        scores.shape
    )
    print("="*70)




# ==========================================================
# MAIN
# ==========================================================


def main():

    parser=argparse.ArgumentParser()



    parser.add_argument(
        "--train",
        action="store_true"
    )


    parser.add_argument(
        "--test",
        action="store_true"
    )



    parser.add_argument(
        "--container",
        required=True
    )


    parser.add_argument(
        "--dataset",
        default=
        "datasets/celebA/datasetfile_multitask"
    )


    parser.add_argument(
        "--shard",
        type=int,
        required=True
    )


    parser.add_argument(
        "--label",
        default="0"
    )


    parser.add_argument(
        "--slices",
        type=int,
        default=1
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )


    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3
    )


    parser.add_argument(
        "--optimizer",
        default="adam"
    )


    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.3
    )


    parser.add_argument(
        "--chkpt_interval",
        type=int,
        default=5
    )



    args=parser.parse_args()



    os.makedirs(
        f"containers/{args.container}/cache",
        exist_ok=True
    )


    os.makedirs(
        f"containers/{args.container}/times",
        exist_ok=True
    )


    os.makedirs(
        f"containers/{args.container}/outputs",
        exist_ok=True
    )



    if args.train:

        train(args)



    if args.test:

        test(args)





if __name__=="__main__":

    main()