import torch
from torch import nn, optim

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import HOMEWORK_DIR, save_model, Detector


# Training Function
def train(
        num_epochs=10,
        learning_rate=1e-3,
        batch_size=32,
        train_data_path="../road_data/train",
        val_data_path="../road_data/val",
):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Detector().to(device)

    # Load datasets
    train_loader = load_data(train_data_path, transform_pipeline="aug", return_dataloader=True, batch_size=batch_size, shuffle=True)
    val_loader = load_data(val_data_path, transform_pipeline="default", return_dataloader=True, batch_size=batch_size, shuffle=False)

    segmentation_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize metrics
    best_miou = 0.0
    val_metric = DetectionMetric(num_classes=3)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Training loop
        for batch in train_loader:
            images = batch['image'].to(device)
            seg_labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)

            optimizer.zero_grad()

            logits, depth_preds = model(images)

            # Compute segmentation loss
            seg_loss = segmentation_loss_fn(logits, seg_labels)

            # Compute depth loss
            depth_loss = depth_loss_fn(depth_preds.squeeze(1), depth_labels)

            # Combine losses
            loss = seg_loss + 0.5 * depth_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                seg_labels = batch['track'].to(device)
                depth_labels = batch['depth'].to(device)

                seg_logits, depth_pred = model(images)
                preds = seg_logits.argmax(dim=1)
                val_metric.add(preds, seg_labels, depth_pred, depth_labels)

        # Compute validation metrics
        metrics = val_metric.compute()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, "
              f"mIoU: {metrics['iou']:.4f}, Depth Error: {metrics['abs_depth_error']:.4f}")

        # Save best model based on mIoU
        if metrics['iou'] > best_miou:
            best_miou = metrics['iou']
            output_path = HOMEWORK_DIR / "detector.th"
            save_model(model)
            print(f"Saved best model with mIoU: {best_miou:.4f}")


# Utility Functions


# Main execution
if __name__ == "__main__":
    train()