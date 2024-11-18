import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

def train_model(model, train_loader, test_loader, device, max_epochs, lr, warmup_ratio):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * max_epochs
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(total_steps * warmup_ratio), T_mult=1, eta_min=0)

    best_loss = np.inf
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} 수행 중")
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            print(f"Validation loss improved from {best_loss:.4f} to {avg_loss:.4f}. 체크포인트를 저장합니다.")
            best_loss = avg_loss
            torch.save(model.state_dict(), '/content/drive/MyDrive/hk-final/best_model.pt')
