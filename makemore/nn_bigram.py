import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

visualize_probs = False
visualize_loss = True

if __name__ == "__main__":
    # load the data
    words = open("makemore/names.txt", "r").read().splitlines()

    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    # create the dataset
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print("Number of examples: ", num)

    # initialize the 'network'
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)

    batches = torch.arange(0, 100, 1)
    losses = []

    xenc = F.one_hot(
        xs, num_classes=27
    ).float()  # input to the network: one-hot encoding

    for _ in batches:
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, allegedly equivalent to N from symbolic bigram
        probs = counts / counts.sum(
            1, keepdims=True
        )  # probabilities for next character
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
        losses.append(loss.item())

        # backward pass
        W.grad = None  # set to zero the gradient
        loss.backward()

        # update
        W.data += -50 * W.grad

        print(f"Batch {len(losses)}: loss = {loss.item()}")

    # visualize the logits matrix
    if visualize_probs:
        visualize_probs()

    # visualize the loss
    if visualize_loss:
        visualize_loss()


def visualize_probs():
    L = (torch.eye(27) @ W).exp()
    P = L / L.sum(1, keepdims=True)

    labels = itos.values()
    fig, ax = plt.subplots(figsize=(10, 10))  # Create a plot with 8x8 in size
    cax = ax.imshow(P.detach().numpy(), cmap="viridis")  # Display color-coded matrix
    print(P.shape)
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


def visualize_loss():
    plt.plot(losses)
    # Annotate plot wit final loss value
    plt.annotate(
        f"Final loss: {round(losses[-1], 2)}",
        xy=(len(losses), losses[-1]),
        xytext=(len(losses) - 10, losses[-1] + 0.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )
    plt.savefig("makemore/exercises/lecture_1/nn_bigram_loss.png")
    plt.show()


def sample():
    g = torch.Generator().manual_seed(2147483647)

    for i in range(5):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W  # predict log-counts
            counts = logits.exp()  # counts, equivalent to N
            p = counts / counts.sum(
                1, keepdims=True
            )  # probabilities for next character

            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            if ix == 0:
                break
        print("".join(out))
