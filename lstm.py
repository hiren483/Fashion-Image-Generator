import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

text = "hello"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = sorted(set(text))  # unique letters: ['e', 'h', 'l', 'o']

char_to_idx = { ch:idx for idx, ch in enumerate(vocab) }
idx_to_char = { idx:ch for ch, idx in char_to_idx.items() }

X = [[1, 0, 2],   # 'h', 'e', 'l'
     [0, 2, 2]]   # 'e', 'l', 'l'

Y = [2, 3]        # 'l', 'o'

X = torch.tensor(X)
Y = torch.tensor(Y)

class CharLSTM(nn.Module):
    def __init__(self,vocab_size , embed_dim , hidden_dim):
        super(CharLSTM,self).__init__()
        
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self , x):
        x = self.embedding(x)       # (batch, seq_len) → (batch, seq_len,    embed_dim)
        out, _ = self.lstm(x)       # out: (batch, seq_len, hidden_dim)
        out = self.fc(out[:, -1])   # Only use the last output for prediction
        return out
    
vocab_size = 4        # 'h', 'e', 'l', 'o'
embed_dim = 8         # small, enough to learn difference
hidden_dim = 32       # memory for tracking 3 chars
num_layers = 1        # one LSTM is enough
learning_rate = 0.01
num_epochs = 2
#defining model
model = CharLSTM(vocab_size , embed_dim , hidden_dim).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)                 # batch input
    loss = criterion(output, Y)      # batch target
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        
input = [char_to_idx['h'], char_to_idx['e'],char_to_idx['l']]
input = torch.tensor(input).unsqueeze(0)  
input = input.to(device)

output = model(input)         
pred = torch.argmax(output)   

print("Next char:", idx_to_char[pred.item()])
