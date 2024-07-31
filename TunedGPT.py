import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer import encode, decode
import time, pickle
import random

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
with open('changes.pickle', 'rb') as f:
    changes = pickle.load(f)
vocab_size = len(changes)
data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 20000
eval_interval = 500
learning_rate = 1e-4
n_embd = 384
n_head = 12
n_layers = 12
dropout = 0.2
block_size = 128
batch_size = 64
output_number = 5
outputtxt_number = 100

train_newlines = []
for i in range (train_data.size(0) - block_size - 2):
    if train_data[i] == 0:
        train_newlines.append(i)

test_newlines = []
for i in range (test_data.size(0) - block_size - 2):
    if test_data[i] == 0:
        test_newlines.append(i)

def get_batch(split):
    data = train_data if split == 'train' else test_data
    newlines = train_newlines if split == 'train' else test_newlines
    ix = torch.tensor(random.sample(newlines, batch_size))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), 
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd), #whats this 4 again?
                                 nn.Dropout(dropout))
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        # normalize to make sure variance doesn't get too high
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
xb, yb = get_batch('train')
logits, loss = m(xb, yb)
print(loss.item())
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device=device), max_new_tokens=100)[0].tolist()))
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for steps in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    if steps % eval_interval == eval_interval-1:
        print(steps, loss.item(), time.time())
        for i in range (output_number):
            sample_out = decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device=device), max_new_tokens=128)[0].tolist())
            print(sample_out)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open("Chess_Transformer_Model_Architecture", 'w') as file:
    file.write(model)

with open("output.txt", 'w') as file:
    for i in range (outputtxt_number):
        file.write(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long, device=device), max_new_tokens=128)[0].tolist()))

torch.save(model.state_dict(), 'Chess_Transformer_Model')
print(model)