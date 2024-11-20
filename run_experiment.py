import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.KET5 import T5ConditionalGeneration
from data_provider.dataset import T5SummaryDataset
from trainer.train import train_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper = T5ConditionalGeneration().to(device)
    tokenizer = model_wrapper.tokenizer

    df = pd.read_csv(".tmp/preprocessed_card_data.csv") # DB에서 가져와 함

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['card_type'])

    train_dataset = T5SummaryDataset(train_data, tokenizer, max_len=1024)
    test_dataset = T5SummaryDataset(test_data, tokenizer, max_len=1024)
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=2)

    # 모델 학습
    train_model(model_wrapper, train_loader, test_loader, device, max_epochs=10, lr=1e-4, warmup_ratio=0.1)