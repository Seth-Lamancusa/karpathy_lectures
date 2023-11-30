import torch
import matplotlib.pyplot as plt

visualize_N = True
visualize_P = False
if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()

    # Gets list of characters from words dataset and generates dictionaries for easy conversion between characters and indices
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    print(f"Characters: {chars}")

    # Generate adjacency matrix N
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # Visualize adjacency matrix N
    if visualize_N:
        labels = itos.values()
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a plot with 8x8 in size
        cax = ax.imshow(N, cmap="viridis")  # Display color-coded matrix
        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                ax.text(
                    j,
                    i,
                    round(N[i, j].item(), 2),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=6,
                )
        ax.set_xticks(torch.arange(N.shape[1]))
        ax.set_yticks(torch.arange(N.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Color Scale")
        plt.show()

    P = (N + 1).float()
    P /= P.sum(dim=1, keepdim=True)

    # Visualize probability matrix P
    if visualize_P:
        labels = itos.values()
        fig, ax = plt.subplots(figsize=(10, 10))  # Create a plot with 8x8 in size
        cax = ax.imshow(P, cmap="viridis")  # Display color-coded matrix
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                ax.text(
                    j,
                    i,
                    round(P[i, j].item(), 2),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=6,
                )
        ax.set_xticks(torch.arange(P.shape[1]))
        ax.set_yticks(torch.arange(P.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Color Scale")
        plt.show()

    g = torch.Generator().manual_seed(2147483647)
    for i in range(20):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))

    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob).item()
            log_likelihood += logprob
            n += 1

    print(f"normalized negative log likelihood: {-log_likelihood / n:.4f}")

    # Inference

    name = []
    name[0] = ["."]
    while not name[-1] == ".":
        p = P[stoi[name[-1]]]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        name.append(itos[ix])
