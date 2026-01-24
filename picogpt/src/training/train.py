import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import TiktokenTokenizer
from model import MiniGPT


BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.block_size])
        y = torch.tensor(self.tokens[idx + 1 : idx + self.block_size + 1])
        return x, y


# Load data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = TiktokenTokenizer()
tokens = tokenizer.encode(text)

dataset = TextDataset(tokens, BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MiniGPT(vocab_size=tokenizer.vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "mini_gpt.pt")
print("Training complete.")
