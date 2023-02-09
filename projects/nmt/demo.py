import torch as th
import sys

from src.config import Seq2SeqConfig, TrainingConfig
from src.decoding import greedy_decode
from src.model import load_model_from_config
from src.tokenization import load_tokenizer_from_file
from src.utils import parse_config_file
from src.hooks import PDBExceptionHook

# sys.excepthook = PDBExceptionHook()

if __name__ == "__main__":
    train_conf: TrainingConfig = parse_config_file("./config/training.json", TrainingConfig)
    model_config: Seq2SeqConfig = parse_config_file("./config/seq2seq.json", Seq2SeqConfig)

    device = th.device('cpu')

    src_tokenizer = load_tokenizer_from_file(train_conf.src_vocab)
    tgt_tokenizer = load_tokenizer_from_file(train_conf.tgt_vocab)

    src_vocab = src_tokenizer.vocab_size
    tgt_vocab = tgt_tokenizer.vocab_size

    log_every_n_steps = train_conf.log_every_n_steps

    model = load_model_from_config(src_vocab, tgt_vocab, model_config, model_file='./data/model_weights/demo.pt',
                                   device=device)

    start_symbol = src_tokenizer.bos_token_id
    end_symbol = src_tokenizer.eos_token_id
    padding_symbol = src_tokenizer.pad_token_id
    max_len = 20

    text = ["I want this text to be translated"]
    input_ids = src_tokenizer(text)
    mask = th.ones(input_ids.shape).type_as(input_ids.data)

    output = greedy_decode(model, input_ids, mask, max_len, start_symbol, end_symbol, padding_symbol)
    print(output)
    print(tgt_tokenizer.decode(output))
