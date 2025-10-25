# Character-Level LSTM for Text Generation

A PyTorch implementation of a multi-layer Long Short-Term Memory (LSTM) network for character-level text generation

## Overview

This project implements a character-level language model using LSTM architecture. The model is trained on Shakespeare's text corpus and learns to generate similar dramatic text character by character. LSTMs improve upon vanilla RNNs by addressing the vanishing gradient problem through gating mechanisms.

## ✨ Advantages of LSTMs

1. **Long-Term Dependencies**: LSTMs can effectively learn dependencies across hundreds of timesteps, unlike vanilla RNNs that struggle beyond 10-20 timesteps.

2. **Gating Mechanism**: Three gates (forget, input, output) allow the model to selectively remember or forget information, providing fine-grained control over information flow.

3. **Cell State**: Separate cell state acts as a "memory highway" that allows gradients to flow unchanged, mitigating the vanishing gradient problem.

4. **Better Gradient Flow**: The additive cell state update (rather than multiplicative) helps preserve gradient magnitude during backpropagation through time.

5. **Improved Training Stability**: More stable and reliable training compared to vanilla RNNs, with less sensitivity to hyperparameters.

## ⚠️ Disadvantages of LSTMs

1. **Computational Complexity**: ~4x more parameters than vanilla RNNs (due to three gates), leading to slower training and inference.

2. **Memory Requirements**: Requires storing both hidden state and cell state, increasing memory footprint.

3. **Sequential Computation**: Still processes sequences step-by-step like RNNs, cannot be parallelized like transformers.

4. **Training Time**: Significantly slower than vanilla RNNs due to increased computational complexity per timestep.

## 🏗️ Implementation Details

### Architecture

The implementation consists of four main components:

1. **ForgetGate**: Controls what information to discard from cell state:
   ```
   f_t = sigmoid(W_xh * x_t + W_hh * h_{t-1} + b_f)
   ```

2. **InputGate**: Determines what new information to add to cell state:
   ```
   i_t = sigmoid(W_xhi * x_t + W_hhi * h_{t-1} + b_i)
   c̃_t = tanh(W_xhc * x_t + W_hhc * h_{t-1} + b_c)
   ```

3. **OutputGate**: Controls what to output based on cell state:
   ```
   o_t = sigmoid(W_xho * x_t + W_hho * h_{t-1} + b_o)
   h_t = o_t * tanh(c_t)
   ```

4. **LSTMCell**: Combines all gates with cell state update:
   ```
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
   ```

### Model Configuration

```python
vocab_size = 65         # Unique characters in dataset
hidden_size = 256       # Hidden state dimension
no_of_layers = 2        # Number of stacked LSTM layers
dropout = 0.1           # Dropout probability between layers
context_len = 100       # Sequence length for training
batch_size = 64         # Training batch size
learning_rate = 0.001   # Adam optimizer learning rate
```

### Dataset

**Source**: `Datasets/shakespear.txt` (from Karpathy's blog, contains 1 million chars)
**Split**: 90% train, 10% validation

## 📊 Training Results

```
Epochs: 5
Best Validation Loss: 1.525215 (Epoch 4)

Epoch 1/20 - Train: 1.659696, Val: 1.605169
Epoch 2/20 - Train: 1.399586, Val: 1.543229
Epoch 3/20 - Train: 1.333020, Val: 1.528535
Epoch 4/20 - Train: 1.293717, Val: 1.525215 ✓ Best Model
Epoch 5/20 - Train: 1.266546, Val: 1.529006
```

## 🎯 Demo Output

```
SEBASTIANIO:
But what you be all understand you so,
And let him be good the news have lived
Not many in one marrs for she--

TRANIO:
Will you! My soul' this words at her wish:
His daughter's Lip, I did you marry her,
Believe him, 'tis strange to put her.

LUCENTIO:
Tranio, a thrusty servant?

PETRUCHIO:
But for what then, here is a man as means,
For she is fear'd haste to woman on the bap
And what painte you the rest, and dead I'll tell him
faults, she'll promise him with a ward.

TRANIO:
As long that I would say it be so.

HORTENSIO:
Thanks, grows so part all encounter ear:
Belive, he do word thee the life in thee
A word to come as fould.

FRIAR PETRUCHIO:
'Tis given me her, I may I discontent
To he have gively for a thousand her:
This hands of all your suitors to be
The first help you what are you no less now.

KATHARINA:
I am sooneth me; whom had you a kind of tresples.

SLY:
For seem as of Lancaster, I live her, as if
they rail them well please awhile within you.

GREMIO:
Sir, sir, I speak about the morning, you may
This time and noble tears you she shall stand me
Sleep might be pleased.
```

## 📚 References

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

