import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import auraloss
import os
import json
import time
import multiprocessing
from tqdm import tqdm

from model import ParametricTCN
from dataset import AudioPotDataset

# Auto-detect GPU (CUDA) for training or fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global Hyperparameters for the TCN Architecture
SEGMENT_LEN = 16384  # Approx 340ms at 48kHz - enough for low-frequency cycles
HIDDEN_CH = 32
NUM_LAYERS = 10

# Loss Function Balancing
# ESR (Error-to-Signal Ratio) focuses on time-domain waveform matching.
# STFT (Short-Time Fourier Transform) focuses on frequency response/spectral match.
TARGET_ALPHA_ESR = 0.1
TARGET_BETA_STFT = 0.9

def train(args):
    os.makedirs("model_weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. DATA PREPARATION
    full_dataset = AudioPotDataset(
        root_dir=args.data_dir,
        segment_len=SEGMENT_LEN,
        taps_per_file=args.taps_per_file
    )

    # 80/20 Split: Training vs Validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Optimize data loading by using multiple CPU cores
    num_workers = max(1, multiprocessing.cpu_count() - 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, # Speeds up transfer to GPU
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 2. MODEL & OPTIMIZER INITIALIZATION
    model = ParametricTCN(version=args.version, num_layers=NUM_LAYERS, hidden_ch=HIDDEN_CH).to(DEVICE)

    # Multi-Resolution STFT Loss is industry standard for audio black-box modeling
    criterion_stft = auraloss.freq.MultiResolutionSTFTLoss().to(DEVICE)
    criterion_esr = auraloss.time.ESRLoss().to(DEVICE)

    # AdamW includes decoupled weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Scheduler: Drops LR when validation loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.lr_patience,
        factor=0.5,
        min_lr=1e-6
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    save_path = os.path.join("model_weights", f"best_model_{args.version}.pth")
    start_time = time.time()

    # 3. TRAINING LOOP
    for epoch in range(args.epochs):
        # ESR WARMUP STRATEGY:
        # We start mostly with STFT loss to get the frequency response right,
        # then gradually introduce ESR to refine the time-domain waveform details.
        warmup_epochs = 10
        current_alpha = TARGET_ALPHA_ESR * min(1.0, (epoch + 1) / warmup_epochs)

        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [{args.version.upper()}] ESR_w={current_alpha:.3f}")

        for x, y, p in pbar:
            x, y, p = x.to(DEVICE), y.to(DEVICE), p.to(DEVICE)

            optimizer.zero_grad()
            y_hat = model(x, p)

            # Combined Loss: Weighted sum of Time (ESR) and Frequency (STFT)
            loss_esr = criterion_esr(y_hat, y)
            loss_stft = criterion_stft(y_hat, y)
            loss = (current_alpha * loss_esr) + (TARGET_BETA_STFT * loss_stft)

            if not torch.isfinite(loss):
                continue

            loss.backward()

            # Gradient Clipping: Prevents "Exploding Gradients" in deep TCN layers
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # 4. VALIDATION PHASE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, p in val_loader:
                x, y, p = x.to(DEVICE), y.to(DEVICE), p.to(DEVICE)
                y_hat = model(x, p)
                v_loss = (current_alpha * criterion_esr(y_hat, y)) + (TARGET_BETA_STFT * criterion_stft(y_hat, y))
                val_loss += v_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        # SAVE BEST MODEL: Only if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(avg_val_loss)

        # Early stopping: End training if model hasn't improved for a long time
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"--- Early stopping triggered after {epoch+1} epochs ---")
            break

    # 5. EXPORT METRICS
    total_time = time.time() - start_time
    json_path = os.path.join("results", f"results_{args.version}.json")
    with open(json_path, "w") as f:
        json.dump({
            "best_val_loss": best_val_loss,
            "total_time_sec": total_time,
            "avg_epoch_time_sec": total_time / (epoch + 1),
            "history": history
        }, f)
    print(f"Training finished. Results saved to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--taps_per_file', type=int, default=256) # Samples per file per epoch
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for LR scheduler')
    parser.add_argument('--early_stop_patience', type=int, default=300, help='Patience for early stopping')

    parser.add_argument('--data_dir', type=str, default='../DATASET')
    args = parser.parse_args()

    train(args)
