import wandb
import torch as th
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from src.model import create_model_from_config, create_optimizer_from_config
from src.parts import LabelSmoothing, SimpleLossCompute
from src.training import training_step, validation_step
from src.config import Seq2SeqConfig, OptimizerConfig, TrainingConfig
from src.utils import extract_sentences, DataCollator, parse_config_file


if __name__ == '__main__':
    train_conf: TrainingConfig = parse_config_file("./config/training.json", TrainingConfig)
    model_config: Seq2SeqConfig = parse_config_file("./config/seq2seq.json", Seq2SeqConfig)
    optim_config: OptimizerConfig = parse_config_file("./config/optimizer.json", OptimizerConfig)

    experim_conf = {**train_conf.dict(), **model_config.dict(), **optim_config.dict()}

    experim = wandb.init(project="machine-translation", entity="flursky", tags=[train_conf.dataset], config=experim_conf)

    device = th.device(train_conf.device)

    src_tokenizer = AutoTokenizer.from_pretrained(train_conf.src_tokenizer)
    tgt_tokenizer = AutoTokenizer.from_pretrained(train_conf.tgt_tokenizer)

    dataset = load_dataset(train_conf.dataset, train_conf.dataset_name)

    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    test_dataset = dataset['test']

    tokenized_train_data = train_dataset.map(extract_sentences, batch_size=256) \
        .remove_columns('translation') \
        .with_format("torch")

    tokenized_valid_data = train_dataset.map(extract_sentences, batch_size=256) \
        .remove_columns('translation') \
        .with_format("torch")

    tokenized_test_data = train_dataset.map(extract_sentences, batch_size=256) \
        .remove_columns('translation') \
        .with_format("torch")

    assert src_tokenizer.pad_token_id == tgt_tokenizer.pad_token_id, "the two tokenizers must have the same padding idx"
    pad_token = src_tokenizer.pad_token_id

    data_collator = DataCollator(src=train_conf.src, tgt=train_conf.tgt, src_tokenizer=src_tokenizer,
                                 tgt_tokenizer=tgt_tokenizer, pad_token=pad_token)

    train_loader = DataLoader(tokenized_valid_data, batch_size=train_conf.train_batch_size, collate_fn=data_collator)
    valid_loader = DataLoader(tokenized_valid_data, batch_size=train_conf.valid_batch_size, collate_fn=data_collator,
                              shuffle=False)
    test_loader = DataLoader(tokenized_test_data, batch_size=train_conf.valid_batch_size, collate_fn=data_collator,
                             shuffle=False)

    src_vocab = src_tokenizer.vocab_size
    tgt_vocab = tgt_tokenizer.vocab_size

    log_every_n_steps = train_conf.log_every_n_steps

    model = create_model_from_config(src_vocab, tgt_vocab, model_config, device=device)
    optimizer, scheduler = create_optimizer_from_config(optim_config, model)

    criterion = LabelSmoothing(model_config.d_model, padding_idx=pad_token, smoothing=train_conf.smoothing_factor)
    loss_fn = SimpleLossCompute(model.generator, criterion)

    for epoch in range(train_conf.epochs):
        running_training_loss = 0.0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            train_loss = training_step(model=model, compute_loss=loss_fn, optimizer=optimizer,
                                       scheduler=scheduler, batch=batch, device=device)
            running_training_loss += train_loss

            if batch_idx % log_every_n_steps == 0:
                experim.log({
                    "train/loss": train_loss,
                    "train/running_loss": running_training_loss,
                    "train/last_lr": scheduler.get_last_lr()
                })

        train_loss = running_training_loss / len(train_loader)

        running_score = 0.0
        running_loss = 0.0

        for batch_idx, batch in tqdm(enumerate(valid_loader), desc="Validation", total=len(valid_loader)):
            loss, bleu_score = validation_step(model=model, compute_loss=loss_fn, batch=batch,
                                               device=device, tgt_tokenizer=tgt_tokenizer)
            running_score += bleu_score
            running_loss += loss

            if batch_idx % log_every_n_steps == 0:
                experim.log({
                    "valid/loss": loss,
                    "valid/running_loss": running_loss,
                    "valid/bleu": validation_bleu
                })

        validation_bleu = running_score / len(valid_loader)
        validation_loss = running_loss / len(valid_loader)

        experim.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_bleu": validation_bleu
        })

        th.save(model.state_dict(), f"./data/model_weights/{epoch}-seq2seq.pt")

