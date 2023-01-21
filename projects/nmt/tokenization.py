from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset


def map_example(example):
    return {"text": example['translation']['ar']}


if __name__ == "__main__":
    dataset = load_dataset('iwslt2017', 'iwslt2017-ar-en')
    arabic_text_data = dataset.map(map_example, batch_size=128)\
        .remove_columns("translation")

    def batch_iterator(batch_size=512):
        for i in range(0, len(arabic_text_data), batch_size):
            yield arabic_text_data['train'][i : i+batch_size]['text']

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'])

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(arabic_text_data))

    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ('[SOS]', tokenizer.token_to_id('[SOS]')),
            ('[EOS]', tokenizer.token_to_id('[EOS]'))
        ]
    )

    tokenizer.enable_padding(pad_token='[PAD]', pad_id=tokenizer.token_to_id('[PAD]'))
    tokenizer.enable_truncation(max_length=10000)

    tokenizer.save("./data/tokenizers/ar-iwslt2017")
