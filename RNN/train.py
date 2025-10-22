import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import CharStreamDataset, save_checkpoint, load_checkpoint
from model import CharRNN
from datetime import datetime
import os


# Load data
with open("../Datasets/shakespear.txt", "r") as f:
    text = f.read()
data_stream = list(text)


# config hyperparameters...
device = "mps"
num_epochs = 20
context_len = 100
vocab_size = len(set(data_stream))
batch_size = 64
hidden_size = 512
no_of_layers = 3
learning_rate = 0.0001


# Load data & define model
loader = DataLoader(
    CharStreamDataset(data_stream, context_len, vocab_size, device),
    batch_size=batch_size, drop_last=True,
)
model = CharRNN(vocab_size, hidden_size, no_of_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

# Load checkpoint if it exists
checkpoint_path = "best_char_rnn.pt"
if os.path.exists(checkpoint_path):
    epoch, loss = load_checkpoint(model, optimizer, checkpoint_path, device)
    print(f"Loaded model from epoch {epoch} with loss {loss}")
    best_loss = loss
else:
    print("No checkpoint found. Starting training from scratch.")
    best_loss = float("inf")


log_file = "training_log.txt"
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
    total_loss = 0
    hidden = None
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred, hidden = model(x, hidden)

        # convert one-hot target to indices
        y_reshaped = y.argmax(dim=-1).reshape(-1)
        pred_reshaped = pred.reshape(-1, vocab_size)

        #(N, C), (N,)
        loss = criterion(pred_reshaped, y_reshaped)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        # DETACH to prevent backprop through entire epoch
        # this prevents gradient flow to previous batches 
        if hidden is not None:
            hidden = [h.detach() for h in hidden]

        total_loss += loss.item()
        if batch_idx % 1000 == 0:
            print(f"epoch - {epoch} batch {batch_idx} Loss : {loss.item():.4}")
    total_loss = total_loss / len(loader)
    print(f"{epoch + 1} Total Loss : {total_loss:.4}")

    # Log epoch results
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.6f}\n")

    if total_loss < best_loss:
        best_loss = total_loss
        save_checkpoint(model, optimizer, epoch, best_loss, "best_char_rnn.pt")

        # Log best model save
        with open(log_file, "a") as f:
            f.write(
                f"  >>> NEW BEST MODEL - Epoch {epoch + 1}, Loss: {best_loss:.6f}\n"
            )

# Log training completion
with open(log_file, "a") as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Final best loss: {best_loss:.6f}\n")
    f.write(f"{'='*60}\n\n")
