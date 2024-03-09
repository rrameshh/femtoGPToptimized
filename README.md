# femtoGPT

GPT implementation in pure Python.
** added autovectorization + compiled code in C

inpired by blog posts: https://bernsteinbear.com/blog/compiling-ml-models/
                       
                       https://bernsteinbear.com/blog/vectorizing-ml-models/

You've seen [nanoGPT](https://github.com/karpathy/nanoGPT).

You've seen [picoGPT](https://github.com/jaymody/picoGPT).

Now, imagine you're alone on a desert island 🏝️ with a computer, Python installed and no Internet (you can't do `pip install torch` 🥲).

A crazy idea occurs to you: what if I created a GPT model... from scratch?

Here is the result of this work: femtoGPT, a GPT implementation in pure Python, without any dependencies... and with it's own autograd engine (inspired by [micrograd](https://github.com/karpathy/micrograd/tree/master)).



## Features

- No tensor ❌
- Not fast ❌
- No batch (one sample at a time) ❌
- No GPU support ❌

A pure learning experience.

Enjoy!

## Usage

Install the package:

```bash
git clone https://github.com/cthiriet/femtoGPT
pip install .
```

Train a femto GPT model on a dataset, that is actually...just a string:

```bash
# Train on the "hello world" sentence
python femtogpt/train.py --dataset "hello world"
```

## Tests

```bash
pytest
```
