from pydantic import BaseModel


class Seq2SeqConfig(BaseModel):
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_model: int = 1024
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: int = 0.1
    pad_token: int = 0


class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    warmup_steps: int = 4000
    d_model: int = 1024


class TrainingConfig(BaseModel):
    device: str
    src: str
    tgt: str
    dataset: str
    dataset_name: str
    src_tokenizer: str
    tgt_tokenizer: str
    epochs: int
    train_batch_size: int
    valid_batch_size: int
    num_workers: int
    log_every_n_steps: int
    smoothing_factor: float
