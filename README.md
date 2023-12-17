# karpathy_lectures

Repository for personal notes and completion of exercises from the public lectures available on Andrej Karpathy's [YouTube Channel](https://www.youtube.com/@AndrejKarpathy), also viewable at [Karpathy's website](https://karpathy.ai/zero-to-hero.html). I did not/do not intend to complete all the exercises. The ones I've completed or am working now on are enumerated in this README with links to solutions.

## Lectures

### Building micrograd

Per Karpathy's repo, micrograd is "A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes."
* [micrograd repo](https://github.com/karpathy/micrograd)
* [micrograd lecture: NNs and backprop](https://www.youtube.com/watch?v=VMj-3S1tku0)

### Building makemore

Per Karpathy's repo, "makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble."
* [makemore repo](https://github.com/karpathy/makemore)
* [makemore lecture 1: Intro](https://www.youtube.com/watch?v=PaCmpygFfXo)
    - **E01**: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model? (See [my solution](makemore/exercises/lecture_1/e01))
    - **E02**: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see? (In progress)
* [makemore lecture 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
* [makemore lecture 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)
* [makemore lecture 4: Becoming a backprop ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI)
* [makemore lecture 5: WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)

### Building GPT

Per Karpathy's video description, nanogpt is a "Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3."
* [nanogpt repo](https://github.com/karpathy/ng-video-lecture)
* [nanogpt lecture](https://www.youtube.com/watch?v=kCc8FmEb1nY)
