import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

class T5ConditionalGeneration(torch.nn.Module):
    def __init__(self):
        super(T5ConditionalGeneration, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('KETI-AIR/ke-t5-small-ko')
        self.tokenizer = T5TokenizerFast.from_pretrained('KETI-AIR/ke-t5-small-ko')
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        return self.model(
            input_ids=inputs['input_ids'],
            attention_mask=attention_mask,
            labels=inputs['labels'],
            return_dict=True
        )
