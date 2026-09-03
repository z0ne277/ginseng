import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader

from model import MoCoV3Ginseng
from UnsupervisedContrastiveDataset import UnsupervisedContrastiveDataset
from config import load_config


def get_data_loaders(config, image_preprocess):
    train_dataset = UnsupervisedContrastiveDataset(
        config['train_csv'], image_preprocess, use_augment=config.get('use_augment', True)
    )
    val_dataset = UnsupervisedContrastiveDataset(
        config['val_csv'], image_preprocess, use_augment=False
    )
    test_dataset = UnsupervisedContrastiveDataset(
        config['test_csv'], image_preprocess, use_augment=False
    )
    return {
        'train': DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 0),
            drop_last=True,
            pin_memory=config.get('pin_memory', False)
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            drop_last=False,
            pin_memory=config.get('pin_memory', False)
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            drop_last=False,
            pin_memory=config.get('pin_memory', False)
        )
    }


def train_model(model, data_loaders, device, optimizer, scheduler, config):
    best_val_loss = float('inf')
    patience_limit = config['patience']
    early_stop_counter = 0
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    best_model_path = os.path.join(config['checkpoint_dir'], "best_model.pth")
    last_model_path = os.path.join(config['checkpoint_dir'], "last_model.pth")

    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        num_batches = len(data_loaders['train'])
        for batch_idx, (img1, img2) in enumerate(data_loaders['train']):
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            q, k = model(img1, img2)
            loss = model.contrastive_loss(q, k)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.momentum_update_key_encoder()
                model.update_queue(k)
            total_train_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")

        train_loss = total_train_loss / max(1, num_batches)
        print(f"✅ Epoch [{epoch + 1}/{config['num_epochs']}], Train Loss: {train_loss:.4f}")

        val_loss = evaluate_loss(model, data_loaders['val'], device)
        print(f"🧠 Validation Loss: {val_loss:.4f}")
        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"🟢 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(model.state_dict(), last_model_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved at epoch {epoch + 1}, val_loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"⏳ Early stopping counter: {early_stop_counter}/{patience_limit}, best_val_loss: {best_val_loss:.4f}")
        if early_stop_counter >= patience_limit:
            print("⚠ Early stopping triggered.")
            break


def evaluate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    with torch.no_grad():
        for img1, img2 in data_loader:
            img1, img2 = img1.to(device), img2.to(device)
            q, k = model(img1, img2)
            loss = model.contrastive_loss(q, k)
            total_loss += loss.item()
    return total_loss / max(1, num_batches)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MoCo-CBAM')
    parser.add_argument('--config', type=str, default=None, help='Optional config JSON path')
    parser.add_argument('--override', action='append', default=[], help='Override key=value (repeatable)')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config('train', config_path=args.config, kv_overrides=args.override)

    image_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    device = torch.device("cuda" if config['use_gpu'] else "cpu")
    print(f"🚀 Using device: {device}")

    model = MoCoV3Ginseng(
        feature_dim=config['feature_dim'],
        K=config['K'],
        m=config['m'],
        T=config['T'],
        device=device
    ).to(device)

    data_loaders = get_data_loaders(config, image_preprocess)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))

    scheduler = None
    if str(config.get('lr_scheduler', 'plateau')).lower() == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    train_model(model, data_loaders, device, optimizer, scheduler, config)


if __name__ == "__main__":
    main()
