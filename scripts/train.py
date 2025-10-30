import json
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from src.models import CodeSearchModel
from src.data import load_cosqa_data
from config import (
    MODEL_NAME, LEARNING_RATE, BATCH_SIZE, EPOCHS,
    MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES, OUTPUT_DIR, PLOT_PATH, LOSS_FUNCTION
)

def plot_losses(train_losses, val_losses, save_path):
    n = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:n]
    val_losses = val_losses[:n]
    epochs_range = np.arange(1, n + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, marker='o', label='Train')
    plt.plot(epochs_range, val_losses, marker='o', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Loss per Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

def train():
    train_dataset, val_dataset, _, _ = load_cosqa_data(MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES)
    model = CodeSearchModel(
        MODEL_NAME, 
        lr=LEARNING_RATE, 
        batch_size=BATCH_SIZE, 
        loss_function=LOSS_FUNCTION
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=False)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[early_stop],
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model.model.save(OUTPUT_DIR)
    plot_losses(model.train_epoch_losses, model.val_epoch_losses, PLOT_PATH)
    with open("training_history.json", "w") as f:
        json.dump({
            "train_epoch_losses": model.train_epoch_losses, 
            "val_epoch_losses": model.val_epoch_losses, 
            "epochs": EPOCHS
        }, f)
    return model.train_epoch_losses, model.val_epoch_losses

if __name__ == "__main__":
    train()
