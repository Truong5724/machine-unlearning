import numpy as np

container = "utkface"
label = 100
num_shards = 4
target_shard = 0
r = 100

# Load splitfile
shards = np.load(f"containers/{container}/splitfile.npy", allow_pickle=True)

requests = []

for i in range(num_shards):
    if i == target_shard:
        np.random.seed(42)
        forget = np.random.choice(shards[i], r, replace=False)
        requests.append(forget)
    else:
        requests.append(np.array([], dtype=int))

requests = np.array(requests, dtype=object)

np.save(f"containers/{container}/requestfile:{label}.npy", requests)

print("Created requestfile:", f"requestfile:{label}.npy")