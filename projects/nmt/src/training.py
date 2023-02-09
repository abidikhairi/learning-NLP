import torch


def training_step(model, compute_loss, optimizer, scheduler, batch, device):
    optimizer.zero_grad()

    src, src_mask, tgt, tgt_mask, tgt_y, seq_len = batch(device=device)

    output = model(src, src_mask, tgt, tgt_mask)
    loss = compute_loss(output, tgt, norm=seq_len)

    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


@torch.no_grad()
def validation_step(model, compute_loss, batch, device, tgt_tokenizer, metric):
    src, src_mask, tgt, tgt_mask, tgt_y, seq_len = batch(device=device)

    output = model(src, src_mask, tgt, tgt_mask)
    loss = compute_loss(output, tgt, norm=seq_len)

    output = model.validation_step(src, src_mask, tgt, tgt_mask)

    preds = torch.argmax(output, dim=-1)

    preds = tgt_tokenizer.decode(preds)
    references = tgt_tokenizer.decode(tgt)

    score = metric.compute(predictions=preds, references=references)

    return loss, score['bleu']
