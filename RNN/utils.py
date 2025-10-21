import torch
from torch.utils.data import Dataset, DataLoader

class CharStreamDataset(Dataset):
    def __init__(self, dataset, context_len, vocab_size, device):
        self.context_len = context_len
        self.vocab_size = vocab_size # 26chars(max)
        self.chars = sorted(set(dataset))
        self.char2int = { c: i for i, c in enumerate(self.chars)}
        self.dataset = torch.tensor([self.char2int[c] for c in dataset], device=device)
        self.one_hot_embed = torch.eye(self.vocab_size, device=device)


    def __len__(self,):
        return len(self.dataset) - self.context_len

    def __getitem__(self, idx):
        x = self.one_hot_embed[self.dataset[idx : idx + self.context_len]]
        y = self.one_hot_embed[self.dataset[idx + 1 : idx + 1 + self.context_len]]
        return x, y


def save_checkpoint(model, optimizer, epoch, loss, path="char_rnn.pt"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"✅ Model saved at epoch {epoch} with loss {loss:.4f} → {path}")


def load_checkpoint(model, optimizer=None, path="char_rnn.pt", device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    print(f"✅ Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss
