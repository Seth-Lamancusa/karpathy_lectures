import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

visualize_probs = False
visualize_loss = False

if __name__ == "__main__":
    # load data from training set
    words = open("makemore/train.txt", "r").read().splitlines()

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

    # evaluate on test set
    words = open("makemore/test.txt", "r").read().splitlines()
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

    xenc = F.one_hot(
        xs, num_classes=27
    ).float()  # input to the network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, allegedly equivalent to N from symbolic bigram
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(f"Test loss: {loss.item()}")

    # The test loss is pretty much equal to the training loss, which means that the model is not overfitting.
