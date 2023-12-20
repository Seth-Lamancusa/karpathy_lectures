import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def make_dataset(dataset_path):
    words = open(dataset_path, "r").read().splitlines()

    chars = ["."] + sorted(list(set("".join(words))))
    stoi = {s: i for i, s in enumerate(chars)}

    sstoi = {
        (s1, s2): i
        for i, (s1, s2) in enumerate([(s1, s2) for s1 in chars for s2 in chars])
    }
    itoss = {i: (s1, s2) for (s1, s2), i in sstoi.items()}

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

    xenc = F.one_hot(xs, num_classes=len(itoss)).float()

    return (xenc, ys)


def evaluate_loss(dataset_path, W, smoothing_str):
    xenc, ys = make_dataset(dataset_path)

    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = (
        -probs[torch.arange(len(xenc)), ys].log().mean()
        + smoothing_str * (W**2).mean()
    )
    return loss


if __name__ == "__main__":
    xenc, ys = make_dataset("makemore/train.txt")

    # initialize the 'network'
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((len(xenc[0]), 27), generator=g, requires_grad=True)

    batches = torch.arange(0, 100, 1)
    losses = []

    smoothing_str = torch.arange(0, 0.5, 0.05)

    for _ in batches:
        # forward pass
        loss = evaluate_loss("makemore/train.txt", W, 0.01)
        losses.append(loss.item())

        # backward pass
        W.grad = None  # set to zero the gradient
        loss.backward()

        # update
        W.data += -50 * W.grad

        print(f"Batch {len(losses)}: loss = {loss.item()}")

    test_loss = evaluate_loss("makemore/test.txt", W, 0.01)
    print(f"Test loss: {test_loss.item()}")

    # Test and train loss are way too similar. Dunno why. Gonna sleep on it
