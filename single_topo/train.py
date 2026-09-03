import argparse
from functools import partial
import json
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from UnsupervisedContrastiveDataset import UnsupervisedContrastiveDataset
from config import load_config
from model import ImprovedMoCoV3WithTopoSideline


IMAGE_SUFFIXES_DEFAULT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

config = {}
image_preprocess = None
log_file = None


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2025-11-8 model")
    parser.add_argument("--config", type=str, default=None, help="Optional config JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override key=value")
    return parser.parse_args()


def load_runtime_config():
    args = parse_args()
    cfg = load_config("train", config_path=args.config, kv_overrides=args.override)
    cfg["img_suffix"] = tuple(cfg.get("img_suffix", IMAGE_SUFFIXES_DEFAULT))
    return cfg


def build_preprocess(cfg):
    preprocess_cfg = cfg.get("image_preprocess", {})
    resize = int(preprocess_cfg.get("resize", 224))
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_cfg.get("mean", [0.5, 0.5, 0.5]),
                std=preprocess_cfg.get("std", [0.5, 0.5, 0.5]),
            ),
        ]
    )


def seed_data_worker(worker_id, *, base_seed):
    worker_seed = int(base_seed) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(csv_path, transform, batch_size, shuffle, use_augment, cfg):
    seed = int(cfg.get("seed", 42)) + (1 if shuffle else 2)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = UnsupervisedContrastiveDataset(
        csv_file=csv_path,
        transform=transform,
        use_augment=use_augment,
        use_binarization=cfg.get("use_binarization", False),
        binarization_threshold=cfg.get("binarization_threshold", 128),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        drop_last=shuffle,
        pin_memory=cfg["pin_memory"],
        worker_init_fn=partial(seed_data_worker, base_seed=seed),
        generator=generator,
    )


def get_data_loaders(cfg):
    return {
        "train": build_dataloader(cfg["train_csv"], image_preprocess, cfg["batch_size"], True, True, cfg),
        "val": build_dataloader(cfg["val_csv"], image_preprocess, cfg["batch_size"], False, False, cfg),
        "test": build_dataloader(cfg["test_csv"], image_preprocess, cfg["batch_size"], False, False, cfg),
    }


def create_model(cfg, device):
    return ImprovedMoCoV3WithTopoSideline(
        feature_dim=cfg["feature_dim"],
        topo_dim=cfg["topo_dim"],
        K=cfg["K"],
        m=cfg["m"],
        T=cfg["T"],
        topo_weight=cfg["topo_weight"],
        num_erosion_levels=cfg["num_erosion_levels"],
        erosion_kernel_size=cfg.get("erosion_kernel_size", 3),
        topology_operator=cfg.get("topology_operator", "min"),
        topology_negative_source=cfg.get("topology_negative_source", "queue"),
        use_cbam=cfg.get("use_cbam", True),
        backbone_name=cfg.get("backbone_name", "resnet50"),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        device=device,
    ).to(device)


def create_optimizer_and_scheduler(model, cfg):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler_name = cfg.get("lr_scheduler", "plateau")
    if scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("cosine_T_max", 50),
            eta_min=1e-6,
        )
    else:
        scheduler = None
    return optimizer, scheduler


def evaluate_loss(model, data_loader, device):
    model.eval()
    total = {"visual": 0.0, "topo": 0.0, "total": 0.0}
    num_batches = 0
    with torch.no_grad():
        for img1, img2 in data_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            visual_q, visual_k, topo_q, topo_k = model(img1, img2)
            _, diagnostics = model.contrastive_loss(visual_q, visual_k, topo_q, topo_k)
            total["visual"] += diagnostics["visual_loss"]
            total["topo"] += diagnostics["topo_loss"]
            total["total"] += diagnostics["total_loss"]
            num_batches += 1
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return (
        total["visual"] / num_batches,
        total["topo"] / num_batches,
        total["total"] / num_batches,
    )


def train_epoch(model, train_loader, optimizer, device, cfg, logger, epoch_idx):
    model.train()
    total = {"visual": 0.0, "topo": 0.0, "total": 0.0}
    num_batches = len(train_loader)
    epoch_start = time.time()

    for batch_idx, (img1, img2) in enumerate(train_loader, start=1):
        img1 = img1.to(device)
        img2 = img2.to(device)

        optimizer.zero_grad()
        visual_q, visual_k, topo_q, topo_k = model(img1, img2)
        loss, diagnostics = model.contrastive_loss(visual_q, visual_k, topo_q, topo_k)
        loss.backward()

        if cfg.get("gradient_clip_max_norm", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_max_norm"])

        optimizer.step()
        with torch.no_grad():
            model.momentum_update_key_encoder()
            model.update_queue(visual_k, topo_k)

        total["visual"] += diagnostics["visual_loss"]
        total["topo"] += diagnostics["topo_loss"]
        total["total"] += diagnostics["total_loss"]

        if batch_idx % max(1, num_batches // 5) == 0 or batch_idx == num_batches:
            logger.log(
                f"epoch={epoch_idx + 1}/{cfg['num_epochs']} "
                f"batch={batch_idx}/{num_batches} "
                f"visual={total['visual'] / batch_idx:.4f} "
                f"topo={total['topo'] / batch_idx:.4f} "
                f"total={total['total'] / batch_idx:.4f}"
            )

    return (
        total["visual"] / num_batches,
        total["topo"] / num_batches,
        total["total"] / num_batches,
        time.time() - epoch_start,
    )


def save_training_curves(history, checkpoint_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = list(range(1, len(history["train_total_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_total_loss"], label="train")
    axes[0].plot(epochs, history["val_total_loss"], label="val")
    axes[0].set_title("total")
    axes[0].legend()

    axes[1].plot(epochs, history["train_visual_loss"], label="train")
    axes[1].plot(epochs, history["val_visual_loss"], label="val")
    axes[1].set_title("visual")
    axes[1].legend()

    axes[2].plot(epochs, history["train_topo_loss"], label="train")
    axes[2].plot(epochs, history["val_topo_loss"], label="val")
    axes[2].set_title("topo")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(Path(checkpoint_dir) / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_model(model, data_loaders, device, optimizer, scheduler, cfg, logger):
    history = {
        "train_visual_loss": [],
        "train_topo_loss": [],
        "train_total_loss": [],
        "val_visual_loss": [],
        "val_topo_loss": [],
        "val_total_loss": [],
        "epoch_seconds": [],
    }

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    best_model_path = checkpoint_dir / "best_model.pth"
    last_model_path = checkpoint_dir / "last_model.pth"
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pth"
    history_path = checkpoint_dir / "training_history.json"

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(cfg["num_epochs"]):
        train_visual, train_topo, train_total, epoch_seconds = train_epoch(
            model, data_loaders["train"], optimizer, device, cfg, logger, epoch
        )
        val_visual, val_topo, val_total = evaluate_loss(model, data_loaders["val"], device)

        history["train_visual_loss"].append(train_visual)
        history["train_topo_loss"].append(train_topo)
        history["train_total_loss"].append(train_total)
        history["val_visual_loss"].append(val_visual)
        history["val_topo_loss"].append(val_topo)
        history["val_total_loss"].append(val_total)
        history["epoch_seconds"].append(epoch_seconds)

        logger.log(
            f"epoch={epoch + 1} "
            f"train_total={train_total:.4f} "
            f"val_total={val_total:.4f} "
            f"epoch_seconds={epoch_seconds:.1f}"
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_total)
            else:
                scheduler.step()

        torch.save(model.state_dict(), last_model_path)
        if (epoch + 1) % cfg.get("save_every", 10) == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"model_epoch_{epoch + 1}.pth")

        if val_total < best_val_loss:
            best_val_loss = val_total
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                best_checkpoint_path,
            )
            logger.log(f"new_best epoch={epoch + 1} val_total={val_total:.4f}")
        else:
            early_stop_counter += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_training_curves(history, checkpoint_dir)

        if early_stop_counter >= cfg["patience"]:
            logger.log(f"early_stop triggered at epoch={epoch + 1}")
            break

    return history


def main():
    global config, image_preprocess, log_file
    config = load_runtime_config()
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    image_preprocess = build_preprocess(config)

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training_log.txt"
    log_file.write_text("", encoding="utf-8")
    logger = Logger(log_file)

    use_gpu = bool(config.get("use_gpu", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    logger.log(f"device={device}")
    logger.log(f"seed={seed}")
    logger.log(f"checkpoint_dir={checkpoint_dir}")

    (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    data_loaders = get_data_loaders(config)
    logger.log(
        f"dataset train={len(data_loaders['train'].dataset)} "
        f"val={len(data_loaders['val'].dataset)} "
        f"test={len(data_loaders['test'].dataset)}"
    )

    model = create_model(config, device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)

    started_at = datetime.now()
    history = train_model(model, data_loaders, device, optimizer, scheduler, config, logger)
    ended_at = datetime.now()

    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_visual, test_topo, test_total = evaluate_loss(model, data_loaders["test"], device)
    logger.log(f"test_visual={test_visual:.4f} test_topo={test_topo:.4f} test_total={test_total:.4f}")
    logger.log(f"started_at={started_at.isoformat()}")
    logger.log(f"ended_at={ended_at.isoformat()}")
    logger.log("training_completed")


if __name__ == "__main__":
    main()
