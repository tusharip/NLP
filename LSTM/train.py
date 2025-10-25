import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import CharStreamDataset, save_checkpoint, load_checkpoint
from model import CharLSTM
from datetime import datetime
import os

log_file = "training_log.txt"

# Load data
with open("../Datasets/shakespear.txt", "r") as f:
    text = f.read()

# Create SHARED vocabulary from ALL text
all_chars = sorted(set(text))
char2int = {c: i for i, c in enumerate(all_chars)}


# 90% train , 10% val
split_index = int(len(text) * 0.9)
train_text = list(text[:split_index])
val_text = list(text[split_index:])
# data_stream = list(text)

# config hyperparameters...
device = "mps"
vocab_size = len(all_chars)
num_epochs = 20
context_len = 100
batch_size = 64
hidden_size = 256
no_of_layers = 2
learning_rate = 0.001


# Load data & define model
train_loader = DataLoader(
    CharStreamDataset(train_text, context_len, vocab_size, char2int, device),
    batch_size=batch_size,
    drop_last=True,
)
val_loader = DataLoader(
    CharStreamDataset(val_text, context_len, vocab_size, char2int, device),
    batch_size=batch_size,
    drop_last=True,
)

model = CharLSTM(vocab_size, hidden_size, no_of_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

# Load checkpoint if it exists
checkpoint_path = "best_char_lstm.pt"
if os.path.exists(checkpoint_path):
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path, device)
    print(f"Loaded model from epoch {epoch} with loss {loss}")
    best_loss = loss
else:
    print("No checkpoint found. Starting training from scratch.")
    best_loss = float("inf")


with open(log_file, "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Hyperparameters:\n")
    f.write(f"  - Epochs: {num_epochs}\n")
    f.write(f"  - Context length: {context_len}\n")
    f.write(f"  - Vocab size: {vocab_size}\n")
    f.write(f"  - Batch size: {batch_size}\n")
    f.write(f"  - Hidden size: {hidden_size}\n")
    f.write(f"  - Layers: {no_of_layers}\n")
    f.write(f"  - Learning rate: {learning_rate}\n")
    f.write(f"{'='*60}\n\n")

for epoch in range(num_epochs):
    model.train()
    avg_train_loss = 0
    state = [None, None]
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        pred, state = model(x, state)

        # convert one-hot target to indices
        y_reshaped = y.argmax(dim=-1).reshape(-1)
        pred_reshaped = pred.reshape(-1, vocab_size)

        # (N, C), (N,)
        loss = criterion(pred_reshaped, y_reshaped)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        # DETACH to prevent backprop through entire epoch
        # this prevents gradient flow to previous batches
        for i, s in enumerate(state):
            state[i] = [k.detach() for k in s]
        avg_train_loss += loss.item()
        if batch_idx % 1000 == 0:
            print(f"epoch - {epoch} batch {batch_idx} Loss : {loss.item():.4}")
    avg_train_loss = avg_train_loss / len(train_loader)
    print(f"{epoch + 1} Total Loss : {avg_train_loss:.4}")

    # validation phase
    model.eval()
    val_loss = 0
    state = [None, None]
    with torch.no_grad():
        for x, y in val_loader:
            pred, state = model(x, state)
            y_reshaped = y.argmax(dim=-1).reshape(-1)
            pred_reshaped = pred.reshape(-1, vocab_size)

            loss = criterion(pred_reshaped, y_reshaped)
            val_loss += loss.item()

            for i, s in enumerate(state):
                state[i] = [k.detach() for k in s]

    avg_val_loss = val_loss / len(val_loader)

    # Print epoch summary
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"{'='*60}\n")

    # Log results
    with open(log_file, "a") as f:
        f.write(
            f"Epoch {epoch + 1}/{num_epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}\n"
        )

    # Save best model based on VALIDATION loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        save_checkpoint(model, optimizer, epoch, best_loss, "best_char_lstm.pt")

        with open(log_file, "a") as f:
            f.write(f"  >>> NEW BEST MODEL - Val Loss: {best_loss:.6f}\n")


# Log training completion
with open(log_file, "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Final best loss: {best_loss:.6f}\n")
    f.write(f"{'='*60}\n\n")
