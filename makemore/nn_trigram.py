import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

visualize_probs = True
visualize_loss = False

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
    for k, v in itoss.items()[0:10]:
        print(k, v)

    # create the dataset
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for (ch1, ch2), ch3 in zip(zip(chs, chs[1:]), chs[2:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            ix3 = stoi[ch3]
            xs.append((ix1, ix2))
            ys.append(ix3)
    for x, y in list(zip(xs, ys))[0:10]:
        print(f"({itos[x[0]]}, {itos[x[1]]}): {itos[y]}")
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
        xs, num_classes=num
    ).float()  # input to the network: one-hot encoding
    for _ in batches:
        # forward pass
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, allegedly equivalent to N from symbolic bigram
        probs = counts / counts.sum(
            1, keepdims=True
        )  # probabilities for next character
        loss = -probs[torch.arange(num), ys].log().mean()
        losses.append(loss.item())

    #     # backward pass
    #     W.grad = None  # set to zero the gradient
    #     loss.backward()

    #     # update
    #     W.data += -50 * W.grad

    #     print(f"Batch {len(losses)}: loss = {loss.item()}")

    # # visualize the loss
    # if visualize_loss:
    #     plt.plot(losses)
    #     plt.show()

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
