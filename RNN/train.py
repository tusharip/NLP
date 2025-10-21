import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import CharStreamDataset, save_checkpoint, load_checkpoint
from model import CharRNN


# Load data
lines = open("../Datasets/transformer.txt", "r").readlines()
data_stream = []
for line in lines:
    cleaned_line = line.strip()  # remove leading/trailing whitespace
    for char in cleaned_line:
        data_stream.append(char)


# config hyperparameters...
device = "mps"
num_epochs = 1
context_len = 100
vocab_size = len(set(data_stream))
batch_size = 64
hidden_size = 256
no_of_layers = 3
learning_rate = 0.0001


# Load data & define model
loader = DataLoader(
    CharStreamDataset(data_stream, context_len, vocab_size, device),
    batch_size=batch_size,
)
model = CharRNN(vocab_size, hidden_size, no_of_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load("best_char_rnn.pt", map_location=device))

print(f"vocab_size: {vocab_size}")


best_loss = float("inf")
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred, h = model(x)

        # convert one-hot target to indices
        y_reshaped = y.argmax(dim=-1).reshape(-1)
        pred_reshaped = pred.reshape(-1, vocab_size)

        loss = criterion(pred_reshaped, y_reshaped)
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) 
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 200 == 0:
            print(f"epoch - {epoch} batch {batch_idx} Loss : {loss.item():.4}")
    total_loss = total_loss / len(loader)
    print(f"{epoch + 1} Total Loss : {total_loss:.4}")

    if total_loss < best_loss:
        best_loss = total_loss
        save_checkpoint(model, optimizer, epoch, best_loss, "best_char_rnn.pt")
