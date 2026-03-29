import torch
import torch.nn.functional as F
import argparse
from contextlib import nullcontext

from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    metric = DetectionMetric()

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            pred_track, pred_depth = model.predict(image)
            metric.add(pred_track, track, pred_depth, depth)

    return metric.compute()


def train(
    epochs: int = 30,
    batch_size: int = 24,
    lr: float = 8e-4,
    num_workers: int = 2,
    depth_weight: float = 2.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    seg_class_weights = torch.tensor([0.25, 1.0, 1.0], device=device)

    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline="default",
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    best_score = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(device)
            track = batch["track"].to(device)
            depth = batch["depth"].to(device)

            optimizer.zero_grad()
            amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
            with amp_ctx():
                logits, raw_depth = model(image)
                depth_pred = torch.sigmoid(raw_depth)

                loss_seg = F.cross_entropy(logits, track, weight=seg_class_weights)
                loss_depth = F.l1_loss(depth_pred, depth)
                loss = loss_seg + depth_weight * loss_depth

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        # Prioritize IoU and boundary depth quality.
        val_score = val_metrics["iou"] - 0.5 * val_metrics["abs_depth_error"] - 0.5 * val_metrics["tp_depth_error"]

        print(
            f"[Epoch {epoch + 1:02d}] "
            f"train_loss={avg_train_loss:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_depth={val_metrics['abs_depth_error']:.4f} "
            f"val_tp_depth={val_metrics['tp_depth_error']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_score > best_score:
            best_score = val_score
            output_path = save_model(model)
            print(f"Saved improved model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--depth_weight", type=float, default=2.0)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        depth_weight=args.depth_weight,
    )
