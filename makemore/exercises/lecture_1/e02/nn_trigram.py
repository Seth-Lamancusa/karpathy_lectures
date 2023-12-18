import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

visualize_loss = True

if __name__ == "__main__":
    # load the data
    words = open("makemore/names.txt", "r").read().splitlines()

    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}

    sstoi = {
        (s1, s2): i
        for i, (s1, s2) in enumerate([(s1, s2) for s1 in chars for s2 in chars])
    }
    itoss = {i: (s1, s2) for (s1, s2), i in sstoi.items()}

    # create the dataset
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for (ch1, ch2), ch3 in zip(zip(chs, chs[1:]), chs[2:]):
            ix1 = sstoi[(ch1, ch2)]
            ix2 = stoi[ch3]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print("Number of examples: ", num)
    print(f"xs size: {xs.size()}")
    print(f"ys size: {ys.size()}")
    for i in range(10):
        print(f"{itoss[xs[i].item()]} -> {itos[ys[i].item()]}")
        print(f"{xs[i]} -> {ys[i]}")

    # initialize the 'network'
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((len(itoss), 27), generator=g, requires_grad=True)
    # len(itoss) (729) is the number of classes in the input: number of inputs to each neuron in the input layer
    # 27 is the number of classes in the output: number of neurons in the output layer

    batches = torch.arange(0, 100, 1)
    losses = []

    xenc = F.one_hot(
        xs, num_classes=len(itoss)
    ).float()  # input to the network: one-hot encoding
    print(f"xenc size: {xenc.size()}")  # Expected: [196118, 729]

    for _ in batches:
        # forward pass
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts
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
    words = open("makemore/exercises/lecture_1/e02/test.txt", "r").read().splitlines()
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
        xs, num_classes=len(itoss)
    ).float()  # input to the network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, allegedly equivalent to N from symbolic bigram
    probs = counts / counts.sum(1, keepdims=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
    print(f"Test loss: {loss.item()}")

    # The test loss is significantly worse than the training loss, which could indicate overfitting. It makes
    # some intuitive sense that this is the case for the trigram model, but not the bigram, since there are probably
    # many unseen trigrams in the test set.
