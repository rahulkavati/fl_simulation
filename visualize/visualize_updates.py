import os
import numpy as np
import matplotlib.pyplot as plt

# Folder containing your .npy files
NumpyFolder = "updates/numpy"

# List all .npy files
all_files = [f for f in os.listdir(NumpyFolder) if f.endswith(".npy")]

# Extract client IDs
clients = sorted(set(f.split("_round_")[0] for f in all_files))

# Plot weight updates per client
for client in clients:
    client_files = sorted([f for f in all_files if f.startswith(client)])
    weight_history = []

    for f in client_files:
        arr = np.load(os.path.join(NumpyFolder, f))
        weight_history.append(arr[:-1])  # exclude bias
    weight_history = np.array(weight_history)

    # Plot each weight
    for w_idx in range(weight_history.shape[1]):
        plt.plot(weight_history[:, w_idx], label=f"w{w_idx}")
    plt.title(f"Weight Updates Over Rounds: {client}")
    plt.xlabel("Round")
    plt.ylabel("Weight Delta")
    plt.legend()
    plt.show()

# Plot bias updates
plt.figure()
for client in clients:
    bias_history = []
    client_files = sorted([f for f in all_files if f.startswith(client)])
    for f in client_files:
        arr = np.load(os.path.join(NumpyFolder, f))
        bias_history.append(arr[-1])  # last element is bias
    plt.plot(bias_history, marker='o', label=f"{client} bias")
plt.title("Bias Updates Over Rounds")
plt.xlabel("Round")
plt.ylabel("Bias Delta")
plt.legend()
plt.show()
