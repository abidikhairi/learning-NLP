import torch
import copy


def learning_rate(step, warmup_steps, d_model):
    step = 1 if step == 0 else step
    rl = (d_model **  -0.5)  * min(step ** -0.5, step * warmup_steps ** (-1.5))
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
    def __init__(self, src: str, tgt: str, src_tokenizer, tgt_tokenizer):
        self.src = src
        self.tgt = tgt
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer

    def __call__(self, features, **kwargs):
        src_sentences = []
        tgt_sentences = []

        for feature in features:
            src_sentences.append(feature[self.src])
            tgt_sentences.append(feature['en'])

        src = self.src_tokenizer(src_sentences, padding=True, truncation=True, return_tensors="pt")
        tgt = self.tgt_tokenizer(tgt_sentences, padding=True, truncation=True, return_tensors="pt")
        tgt['input_ids'] = tgt['input_ids'][:, :-1]
        tgt['attention_mask'] = tgt['attention_mask'][:, :-1]

        tgt_y = tgt['input_ids'][:, 1:]
        return {
            'src': src,
            'tgt': tgt,
            'tgt_y': tgt_y
        }


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

    return src, src_mask, tgt, tgt_mask
