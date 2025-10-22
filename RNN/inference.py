import torch
import torch.nn.functional as F
from utils import load_checkpoint
from model import CharRNN


@torch.no_grad()
def generate(
    model, seed_text, char2int, int2char, vocab_size, gen_len=50, temperature=0.5, device="cuda"
):
    model.eval()
    input_idx = [char2int[c] for c in seed_text]
    input_tensor = torch.zeros(1, len(input_idx), vocab_size, device=device)
    for i, idx in enumerate(input_idx):
        input_tensor[0, i, idx] = 1.0

    hidden = None
    generated = list(seed_text)

    # Warm up model with seed text
    out, hidden = model(input_tensor, hidden)
    next_input = input_tensor[:, -1:, :]

    for _ in range(gen_len):
        out, hidden = model(next_input, hidden)
        probs = F.softmax(out[:, -1, :] / temperature, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = int2char[next_idx]
        generated.append(next_char)

        next_input = torch.zeros(1, 1, vocab_size, device=device)
        next_input[0, 0, next_idx] = 1.0

    return "".join(generated)


if __name__ == "__main__":
    device = "mps"
    vocab_size = 65
    hidden_size = 512
    no_of_layers = 3

    model = CharRNN(vocab_size, hidden_size, no_of_layers).to(device)
    epoch, loss = load_checkpoint(model, None, "best_char_rnn.pt", device)


    # Load data
    with open("../Datasets/shakespear.txt", "r") as f:
        text = f.read()
    data_stream = list(text)
    chars = sorted(set(data_stream))
    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for i, c in enumerate(chars)}

    seed_text = "SEBASTIAN"
    generated_text = generate(
        model, seed_text, char2int, int2char, vocab_size, gen_len=2000, temperature=0.8, device=device
    )
    print("Generated text:", generated_text)
