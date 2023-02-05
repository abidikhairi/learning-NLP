import argparse
import torch as th
import pandas as pd
from typing import List
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator, Vocab


class Tokenizer(th.nn.Module):
    def __init__(self, vocab_file):
        super(Tokenizer, self).__init__()

        self.word2idx = Vocab(th.load(vocab_file))
        self.vocab_size = len(self.word2idx.vocab)

    def prepare_tokens(self, tokens: List[str]):
        ret = []
        sos_token = self.word2idx['<sos>']
        eos_token = self.word2idx['<eos>']

        for token_list in tokens:
            token_list = [sos_token] + token_list + [eos_token]
            ret.append(token_list)

        return ret

    def encode(self, text: List[str]) -> th.LongTensor:
        pad_token_id = self.word2idx['<pad>']

        tokens = list(map(word_tokenize, text))
        token_ids = list(map(self.word2idx, tokens))
        token_ids = self.prepare_tokens(token_ids)
        token_ids = list(map(th.LongTensor, token_ids))

        token_ids = pad_sequence(token_ids, batch_first=True, padding_value=pad_token_id)
        return token_ids

    def forward(self, inputs: List[str]):
        input_ids = self.encode(inputs)

        return input_ids


def yield_tokens(df: pd.DataFrame):
    for _, text in df.itertuples():
        yield word_tokenize(text=text)


def build_tokenizer(args):
    input_file = args.input_file
    output_file = args.output_file
    lang = args.language

    data = pd.read_csv(input_file)
    data_lang = data[[lang]]

    special_tokens = [
        '<sos>', '<pad>', '<unk>', '<eos>'
    ]

    vocab = build_vocab_from_iterator(yield_tokens(data_lang), specials=special_tokens, special_first=True)
    vocab.set_default_index(vocab['<unk>'])

    th.save(vocab, output_file)


def load_tokenizer_from_file(vocab_file: str) -> Tokenizer:
    return Tokenizer(vocab_file=vocab_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build Language specific tokenizer')

    parser.add_argument('--input-file', required=True, type=str)
    parser.add_argument('--language', required=True, type=str)
    parser.add_argument('--output-file', required=True, type=str)

    args = parser.parse_args()

    build_tokenizer(args)
