import pydantic
import torch
import copy


def learning_rate(step, warmup_steps, d_model):
    step = 1 if step == 0 else step
    rl = (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** (-1.5))
    return rl


def extract_sentences(example):
    return {'en': example['translation']['en'], 'fr': example['translation']['fr']}


def move(*args, **kwargs):
    device = kwargs.get('device', 'cpu')
    tensors = []
    for tensor in args:
        tensors.append(tensor.to(device))

    return tensors


class DataCollator(object):
    def __init__(self, src: str, tgt: str, src_tokenizer, tgt_tokenizer):
        self.src = src
        self.tgt = tgt
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer

        self.pad_token = self.src_tokenizer.word2idx['<pad>']

    def __call__(self, features, **kwargs):
        src_sentences = []
        tgt_sentences = []

        for feature in features:
            src_sentences.append(feature[self.src])
            tgt_sentences.append(feature[self.tgt])

        src = self.src_tokenizer(src_sentences)
        tgt = self.tgt_tokenizer(tgt_sentences)

        batch = { 'src': src, 'tgt': tgt }

        return Batch(batch=batch, pad_token=self.pad_token)


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (size, size)

    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def clones(module: torch.nn.Module, n: int = 2):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def prepare_inputs(src, tgt, pad_id):
    diff = abs(tgt.size(1) - src.size(1))

    if diff > 0:
        if tgt.size(1) > src.size(1):
            src = torch.nn.functional.pad(src, (0, diff), mode="constant", value=pad_id)
        else:
            tgt = torch.nn.functional.pad(tgt, (0, diff), mode="constant", value=pad_id)

    seq_len = tgt.size(1)
    tgt_y = torch.roll(tgt, shifts=1, dims=1)

    return src, tgt, tgt_y, seq_len


class Batch:
    def __init__(self, batch: dict, pad_token: int = 0):
        self.src = batch['src']
        self.tgt = batch['tgt']

        self.pad_token = pad_token

    def __call__(self, *args, **kwargs):
        device = kwargs.get('device', 'cpu')

        src, tgt, tgt_y, seq_len = prepare_inputs(self.src, self.tgt, pad_id=self.pad_token)

        batch_size = src.size(0)

        src_mask = torch.ones(src.shape).type_as(src.data)
        src_mask[src == self.pad_token] = 0
        src_mask = src_mask.unsqueeze(2).bool()
        tgt_mask = subsequent_mask(seq_len).repeat(batch_size, 1, 1)

        src, src_mask, tgt, tgt_mask = move(src, src_mask, tgt, tgt_mask, device=device)

        return src, src_mask, tgt, tgt_mask, tgt_y, seq_len


def parse_config_file(path, _class) -> pydantic.BaseModel:
    return pydantic.parse_file_as(_class, path)
