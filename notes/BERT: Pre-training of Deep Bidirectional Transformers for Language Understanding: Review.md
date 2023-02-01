# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## Keywords:
- Language model
- Bidirectional representation
- Fine-tuning

## Section 0: Abstract
**BERT** is a language model, based on the Transformer architecture. It can be fine-tuned to create sota NLP models.
The thought behind **BERT** is to train a large language model in an unsupervised manner. Then, use the pretrained model as a backbone for a more simpler Task specific architecture (downstream task).

- 7.7 % improvement on GLUE benchmark.
- 4.6 % MultiNLI improvement.
- Question answering
    - SQuAD v1 1.5 improvement (_f1_ 93.2).
    - SQuAD v2 5.1 improvement (_f1_ 83.1).

## Section 1: Introduction
-   Pretrained Large Language models are good for NLP downstream tasks (NER, NLI, Sentiment Analysis, etc).
-   There are two ways to apply pretrained models to downstream NLP tasks.
    -   _Feature-based_: includes the pretrained architecture to extract informations, then use a task specific architecture to process the extracted features. 
    -   _Finetuning_: add task specific parameters to the pretrained architecture, then, the whole model is trained on the downstream task (including pretrained parameters).

## Section 3: BERT

BERT framework consists of two steps: _pretraining_ and _finetuning_.
-   During pretraining the model is pretrained in an unsupervised manner.
-   Then, a new model is initialized from the pretrained one (with added task layers) and trained on downstream task (all parameters are optimized).
- Architecture: Transformer Encoder only.
- Text tokenization: [WordPiece Tokenizer](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html)

BERT add a special token `[CLS]` in the first position of the input, the `[CLS]` token is used as an aggregation of the sentence (in sentence level prediction such as sentiment analysis, _etc_)

In tasks requiring input to be a pair of sentences BERT separate the inputs by a special token `[SEP]` and fed the input to the model, as a result, in the attention layer each sentence can attend to the other, hence, BERT achieves a cross attention and the words representation does not only depends on the sentence itself.    

## Pointers
- BERT Loss function is composed of two terms:
    1. Masked Language Model objective
    2. Next Sentence Prediction objective
- Left to Right objectives (Cross Entropy Loss) and Discriminative objectives (Negative sampling) are not sufficient to train models that can represent (therefore understand) text (or words).
