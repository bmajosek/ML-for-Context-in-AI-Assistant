import json
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models import CodeSearchModel
from src.data import load_cosqa_data
from config import (
    MODEL_NAME, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, OUTPUT_DIR, PLOT_PATH
)

def plot_losses(losses, epochs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses, alpha=0.3, color='blue')
    if len(losses) > 20:
        window = 20
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(losses)), smoothed, linewidth=2, label='Smoothed', color='red')
        ax1.legend()
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    steps_per_epoch = len(losses) // epochs
    epoch_losses = [np.mean(losses[i*steps_per_epoch:(i+1)*steps_per_epoch]) for i in range(epochs)]
    
    ax2.bar(range(1, epochs+1), epoch_losses, color='green', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Loss')
    ax2.set_title('Loss per Epoch')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved: {save_path}")

def train():
    train_dataset, val_dataset, _, _ = load_cosqa_data(MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES)
    print(f"Training: {len(train_dataset)} samples, {EPOCHS} epochs")
    
    model = CodeSearchModel(MODEL_NAME, lr=LEARNING_RATE, batch_size=BATCH_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    model.model.save(OUTPUT_DIR)
    print(f"Saved: {OUTPUT_DIR}")
    
    plot_losses(model.losses, EPOCHS, PLOT_PATH)
    
    with open("training_history.json", "w") as f:
        json.dump({"losses": model.losses, "epochs": EPOCHS}, f)
    
    return model.losses

if __name__ == "__main__":
    train()
