# Attention is All You Need

## Part 1:

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
