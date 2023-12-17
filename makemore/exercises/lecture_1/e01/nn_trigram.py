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

    # visualize the loss
    if visualize_loss:
        plt.plot(losses)
        # Annotate plot wit final loss value
        plt.annotate(
            f"Final loss: {round(losses[-1], 2)}",
            xy=(len(losses), losses[-1]),
            xytext=(len(losses) - 10, losses[-1] + 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05),
        )
        plt.savefig("makemore/exercises/lecture_1/nn_trigram_loss.png")
        plt.show()

    # # finally, sample from the 'neural net' model
    # g = torch.Generator().manual_seed(2147483647)

    # for i in range(5):
    #     out = []
    #     ix = 0
    #     while True:
    #         xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    #         logits = xenc @ W  # predict log-counts
    #         counts = logits.exp()  # counts, equivalent to N
    #         p = counts / counts.sum(
    #             1, keepdims=True
    #         )  # probabilities for next character

    #         ix = torch.multinomial(
    #             p, num_samples=1, replacement=True, generator=g
    #         ).item()
    #         out.append(itos[ix])
    #         if ix == 0:
    #             break
    #     print("".join(out))
