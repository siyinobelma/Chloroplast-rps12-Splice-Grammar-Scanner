import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from scripts.model import SplicingCNN
from sklearn.metrics import accuracy_score

def main():
    train_data, train_labels = torch.load(Config.TRAIN_FILE)
    val_data, val_labels = torch.load(Config.VAL_FILE)

    train_set = TensorDataset(train_data, train_labels)
    val_set = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE)

    model = SplicingCNN(Config).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_acc = 0
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
          
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                trues.extend(y.cpu().numpy())
        acc = accuracy_score(trues, preds)
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")
    print(f"Best val acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()
