The following scripts allow to run a sharding experiment on CelebA dataset.

1- First prepare the dataset:
cd datasets/celebA
python prepare_data.py

2- Create a container with a specified number of shards:
bash init.sh 5

3- Train the shards in the container:
bash train.sh 5

4- Compute shard predictions:
bash predict.sh 5

5- Retrieve experimental data as a CSV:
bash data.sh 5

Note: CelebA training will take longer than CIFAR-10 due to larger images (64x64 vs 32x32)
