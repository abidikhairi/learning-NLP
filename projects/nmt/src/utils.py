import torch
import copy


def learning_rate(step, warmup_steps, d_model):
    step = 1 if step == 0 else step
    rl = (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** (-1.5))
    return rl


def extract_sentences(example):
    return {'ar': example['translation']['ar'], 'en': example['translation']['en']}


def move(*args, **kwargs):
    device = kwargs.get('device', 'cpu')
    tensors = []
    for tensor in args:
        tensors.append(tensor.to(device))

    return tensors


class DataCollator(object):
    def __init__(self, src: str, tgt: str, src_tokenizer, tgt_tokenizer, pad_token):
        self.src = src
        self.tgt = tgt
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        self.pad_token = pad_token

    def __call__(self, features, **kwargs):
        src_sentences = []
        tgt_sentences = []

        for feature in features:
            src_sentences.append(feature[self.src])
            tgt_sentences.append(feature['en'])

        src = self.src_tokenizer(src_sentences, padding=True, truncation=True, return_tensors="pt")
        tgt = self.tgt_tokenizer(tgt_sentences, padding=True, truncation=True, return_tensors="pt")

        batch = { 'src': src, 'tgt': tgt }

        return Batch(batch=batch, pad_token=self.pad_token)


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (size, size)

    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


def clones(module: torch.nn.Module, n: int = 2):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def prepare_inputs(src, src_mask, tgt, tgt_mask, pad_id):
    diff = abs(tgt.size(1) - src.size(1))

    if diff > 0:
        if tgt.size(1) > src.size(1):
            src = torch.nn.functional.pad(src, (0, diff), mode="constant", value=pad_id)
            src_mask = torch.nn.functional.pad(src_mask, (0, diff), mode="constant", value=pad_id)
        else:
            tgt = torch.nn.functional.pad(tgt, (0, diff), mode="constant", value=pad_id)
            tgt_mask = torch.nn.functional.pad(tgt_mask, (0, diff), mode="constant", value=pad_id)

    src_mask = src_mask.unsqueeze(2).bool()
    tgt_mask = tgt_mask.unsqueeze(2).bool()
    seq_len = tgt.size(1)
    tgt_y = torch.roll(tgt, shifts=1, dims=1)

    return src, src_mask, tgt, tgt_mask, tgt_y, seq_len


class Batch:
    def __init__(self, batch: dict, pad_token: int = 0):
        self.src = batch['src']['input_ids']
        self.src_mask = batch['src']['attention_mask']
        self.tgt = batch['tgt']['input_ids']
        self.tgt_mask = batch['tgt']['attention_mask']

        self.pad_token = pad_token

    def __call__(self, *args, **kwargs):
        device = kwargs.get('device')
        src, src_mask, tgt, tgt_mask, tgt_y, seq_len = prepare_inputs(self.src, self.src_mask, self.tgt, self.tgt_mask,
                                                                      pad_id=self.pad_token)

        src, src_mask, tgt, tgt_mask = move(src, src_mask, tgt, tgt_mask, device=device)

        return src, src_mask, tgt, tgt_mask, tgt_y, seq_len
