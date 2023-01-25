# Attention is All You Need

### Section 0: Abstract

- Seq2Seq Problems are handled by Encoder-Decoder Architectures
    - Encoders and Decoders are most the time, complex Architectures like RNN and CNN
    - The Encoder is connected to the Decoder through an attention mechanism
- Contributions:
    - New simple model based on attention mechanisms (no reccurence nor convolution)
    - The proposed model is more parallelizable and efficient (short training time)
    - Results:
        - WMT 2014 English-German Translation: 28.4 BLEU
        - WMT 2014 English-French Translation: 41.8 BLEU

### Section 1: Introduction
The first paragraph mention that RNN, specially LSTM and GRU, have achieved sota results in tasks where input data is sequence. Although, such architectures cannot be parallalized due to their nature of processing the input one by one.

In paragraph two it is mentioned that Attention Mechanisms allow models modeling dependencies between inputs and outputs regardless of their positions.

The last paragraph present the proposed model, **The Transformer**, an architecture based solely on attention mechanism (no reccurent layers), which allows to more parallelization.

### Section 2: Background

- Seq2Seq Convolution Based Models:
    1. Extended Neural GPU:
    2. ByteNet:
    3. ConvS2S:

These Architectures are based on convolutions neural networks.

**Self-Attention** is process that _connects_ positions (words) in the same sequence.
Example: "Aujourd'hui il fait beau"

scores are computed between each position and all other positions.

### Section 3: Model Architecture

#### Encoder Decoder Architecture

Encoder input: $x = (x_1, x_2, \dots, x_n)$

Encoder output: $z = (z_1, z_2, \dots, z_n)$

Decoder input: $z = (z_1, z_2, \dots, z_n)$

Decoder output: $y = (y_1, y_2, \dots, y_m)$

$\implies$ The input sequence and output sequence does not need to have the same length.

#### Transformer: Encoder-Decoder Layers

**Encoder Layer**: Composed of two components, _Self-Attention_ and _Position Wise Linear Layer_.

**Decoder Layer**:Composed of three components, _Self-Attention_, _Position Wise Linear Layer_ and _Attention Layer_ between encoder outputs and decoder inputs.

It is worth mentioning that:

- encoder inputs are shifted to the right.
- encoder and decoder are composed of stack of $N = 6$ layers.
- Residual connections is employed, i.e, $x_{layer} = Layer(x) + x$.
- Layer Normalization is applied at each the end of each component, i.e, $h = LayerNorm(Layer(x) + x)$.
- all layers outputs have dimension of $d_{model} = 512$.

#### Transformer: Attention Layer

$\implies$ The attention is a function that maps query and keys to a set of values.

$$ Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V $$

##### Application of Attention
1.  _Encoder-Decoder Attention_: queries come from the Decoder, keys and values are the output of the encoder. The model can learn to align outputs (target) with inputs (source).
2.  _Encoder Attention_: Self Attention is used.
3.  _Decoder Attention_: Self Attention is used, with masking (set to `-inf`) of illegal positions.

In addition of Attention layers, Position wise Feedforward Networks are used (MLP):

$$ FFN(X) = max(0, XW_1 + b_1)W_2 + b_2 $$

#### Positional encoding
$\implies$ inject position (absolute or relative) information to learned word (token) embeddings.


### Section 4: Training

-   WMT 2014 English-German: [link](https://huggingface.co/datasets/wmt14/viewer/de-en/)
-   WMT 2014 English-French: [link](https://huggingface.co/datasets/wmt14/viewer/fr-en/)

**Dropout**: is used and the end of each sub-layer, in addition, it is used (with $p=0.1$) on the sum of embeddings and positional encoding.

#### Label Smoothing

Is a regularization technique, used to tackle the _overconfidence_ problem (a model may predict 0.9 as an outcome while the accuracy is 0.6)

$$ y_{ls} = (1 - \alpha) * y_{hot} + \alpha/K $$
where:
-   $y_{hot}$: is the one hot-encoded label vector.
-   $\alpha$: smoothing factor.
-   $K$: the number of classes.

### Section 5: Results

#### WMT 2014 English-to-German
- Transformer (Big): 28.4 BLEU (_new state-of-the-art_).
    - 3.5 days
    - 8 P100 GPU

#### WMT 2014 English-to-French
- Tansformermer (Big): 41.0 BLEU (_new state-of-the-art_).
    - dropout rate: 0.1
