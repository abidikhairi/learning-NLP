import torch as th
from src.model import Seq2Seq
from src.utils import subsequent_mask

if __name__ == '__main__':
    src_vocab_size = 10000
    tgt_vocab_size = 15000

    model = Seq2Seq(src_vocab_size, tgt_vocab_size)

    src = th.randint(0, 10000, (16, 20))
    src_mask = th.ones(20, 20)

    tgt = th.randint(0, 15000, (16, 30))
    tgt_mask = subsequent_mask(30)

    out = model(src, src_mask, tgt, tgt_mask)
    print(out)
    print(out.shape)
