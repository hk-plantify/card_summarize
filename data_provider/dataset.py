import torch

class T5SummaryDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = df
        self.len = self.docs.shape[0]
        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = [self.pad_index] * (self.max_len - len(inputs))
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = [self.ignore_index] * (self.max_len - len(inputs))
            inputs = inputs + pad
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_text = "summarize: " + str(instance['additional_info'])
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_len, truncation=True)
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(str(instance['benefit_description']), max_length=self.max_len, truncation=True)
        label_ids = self.add_ignored_data(label_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len
