import wandb
import evaluate
import torch as th
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.datasets import TranslationDataset
from src.model import create_model_from_config, create_optimizer_from_config
from src.parts import LabelSmoothing, SimpleLossCompute
from src.training import training_step, validation_step
from src.config import Seq2SeqConfig, OptimizerConfig, TrainingConfig
from src.utils import DataCollator, parse_config_file
from src.tokenization import load_tokenizer_from_file

if __name__ == '__main__':
    train_conf: TrainingConfig = parse_config_file("./config/training.json", TrainingConfig)
    model_config: Seq2SeqConfig = parse_config_file("./config/seq2seq.json", Seq2SeqConfig)
    optim_config: OptimizerConfig = parse_config_file("./config/optimizer.json", OptimizerConfig)

    experim_conf = {**train_conf.dict(), **model_config.dict(), **optim_config.dict()}

    experim = wandb.init(project="machine-translation", entity="flursky", tags=["en-ar"], config=experim_conf)

    device = th.device(train_conf.device)

    src_tokenizer = load_tokenizer_from_file(train_conf.src_vocab)
    tgt_tokenizer = load_tokenizer_from_file(train_conf.tgt_vocab)

    train_data = TranslationDataset('./data/train.csv', source=train_conf.src, target=train_conf.tgt, src_col_idx=0,
                                    tgt_col_idx=1)
    valid_data = TranslationDataset('./data/valid.csv', source=train_conf.src, target=train_conf.tgt, src_col_idx=0,
                                    tgt_col_idx=1)
    test_data = TranslationDataset('./data/test.csv', source=train_conf.src, target=train_conf.tgt, src_col_idx=0,
                                   tgt_col_idx=1)

    pad_token = src_tokenizer.word2idx['<pad>']

    data_collator = DataCollator(src=train_conf.src, tgt=train_conf.tgt, src_tokenizer=src_tokenizer,
                                 tgt_tokenizer=tgt_tokenizer)

    train_loader = DataLoader(train_data, batch_size=train_conf.train_batch_size, collate_fn=data_collator,
                              num_workers=train_conf.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=train_conf.valid_batch_size, collate_fn=data_collator,
                              shuffle=False, num_workers=train_conf.num_workers)
    test_loader = DataLoader(test_data, batch_size=train_conf.valid_batch_size, collate_fn=data_collator,
                             shuffle=False, num_workers=train_conf.num_workers)

    src_vocab = src_tokenizer.vocab_size
    tgt_vocab = tgt_tokenizer.vocab_size

    log_every_n_steps = train_conf.log_every_n_steps

    model = create_model_from_config(src_vocab, tgt_vocab, model_config, device=device)
    optimizer, scheduler = create_optimizer_from_config(optim_config, model)

    criterion = LabelSmoothing(model_config.d_model, padding_idx=pad_token, smoothing=train_conf.smoothing_factor)
    loss_fn = SimpleLossCompute(model.generator, criterion)

    best_blue = 0.0

    for epoch in range(train_conf.epochs):
        running_training_loss = 0.0
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            train_loss = training_step(model=model, compute_loss=loss_fn, optimizer=optimizer,
                                       scheduler=scheduler, batch=batch, device=device)
            running_training_loss += train_loss

            if batch_idx % log_every_n_steps == 0:
                experim.log({
                    "train/loss": train_loss,
                    "train/last_lr": scheduler.get_last_lr()
                })

        train_loss = running_training_loss / len(train_loader)

        running_score = 0.0
        running_loss = 0.0

        metric = evaluate.load('bleu')

        model.eval()
        for batch_idx, batch in tqdm(enumerate(valid_loader), desc="Validation", total=len(valid_loader)):
            loss, bleu_score = validation_step(model=model, compute_loss=loss_fn, batch=batch,
                                               device=device, tgt_tokenizer=tgt_tokenizer, metric=metric)
            running_score += bleu_score
            running_loss += loss

            if batch_idx % log_every_n_steps == 0:
                experim.log({
                    "valid/loss": loss,
                    "valid/bleu": bleu_score
                })

        validation_bleu = running_score / len(valid_loader)
        validation_loss = running_loss / len(valid_loader)

        experim.log({
            "main/epoch": epoch,
            "main/train_loss": train_loss,
            "main/validation_loss": validation_loss,
            "main/validation_bleu": validation_bleu
        })

        if validation_bleu > best_blue:
            best_blue = validation_bleu
            th.save(model.state_dict(), f"./data/model_weights/seq2seq-{epoch}-{best_blue}.pt")
