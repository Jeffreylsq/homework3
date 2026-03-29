import torch
import torch.nn as nn

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data

def train():

    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== model =====
    model = Classifier().to(device)

    # ===== optimizer & loss =====
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # ===== data =====
    train_loader = load_data(
        dataset_path="classification_data/train",
        shuffle=True,
        batch_size=512,
        num_workers=0
    )

    val_loader = load_data(
        dataset_path="classification_data/val",
        shuffle=False,
        batch_size=512,
        num_workers=0
    )

    num_epochs = 2

    for epoch in range(num_epochs):

        # =========================
        # TRAIN
        # =========================
        model.train()

        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # =========================
        # VALIDATION
        # =========================
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[Epoch {epoch}] Val Acc: {acc:.4f}")

    # ===== save =====
    save_model(model)

if __name__ == "__main__":
    train()