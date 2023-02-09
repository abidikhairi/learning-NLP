import torch
import torch.nn.functional as F
from src.utils import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, padding_symbol):
    src = F.pad(src, (0, abs(max_len - src.size(1))), 'constant', padding_symbol)
    src_mask = F.pad(src_mask, (0, abs(max_len - src_mask.size(1))), 'constant', 0)
    memory = model.encode(src, src_mask)

    decoder_input = torch.tensor([[start_symbol]]).type_as(src.data)
    decoder_input = F.pad(decoder_input, (0, abs(max_len - decoder_input.size(1))), 'constant', padding_symbol)
    decoder_mask = subsequent_mask(max_len).long()
    _, idx = torch.where(decoder_input == padding_symbol)
    decoder_mask[idx] = 0
    decoder_mask = decoder_mask.bool()
    decoded_words = [start_symbol]

    for i in range(max_len):
        out = model.decode(memory, src_mask, decoder_input, decoder_mask).squeeze(0)

        probs = torch.softmax(model.generator(out), dim=-1)
        next_word = torch.max(probs).item()

        if next_word == end_symbol:
            break

        decoded_words.append(next_word)
        decoder_input = torch.tensor([decoded_words]).long()

        decoder_input = F.pad(decoder_input, (0, abs(max_len - decoder_input.size(1))), 'constant', padding_symbol)
        decoder_mask = subsequent_mask(max_len).long()
        _, idx = torch.where(decoder_input == padding_symbol)
        decoder_mask[idx] = 0
        decoder_mask = decoder_mask.bool()

    return torch.tensor([decoded_words])
