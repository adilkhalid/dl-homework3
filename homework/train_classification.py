import argparse

import torch.utils.data
from torch import optim, nn

from homework.datasets.classification_dataset import load_data
from homework.models import Classifier, INPUT_MEAN, INPUT_STD, save_model
import torchvision
import torchvision.transforms as transforms


def train():
    model = Classifier()
    learning_rate = 0.001
    momentum = 0.9
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


    train_loader = load_data(
        dataset_path='../classification_data/train',
        transform_pipeline="aug",  # use "default" if no augmentation is needed
        return_dataloader=True,
        num_workers=4,
        batch_size=32,  # or any preferred batch size
        shuffle=True
    )

    # Load validation data without augmentation
    val_loader = load_data(
        dataset_path='../classification_data/val',
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=4,
        batch_size=32,
        shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    model.eval()
    val_loss = 0.0
    val_correct =0
    val_total = 0

    with torch.inference_mode():  # No gradient calculation in eval mode
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

    print("Training completed.")

    save_model(model)


if __name__ == "__main__":
    train()
