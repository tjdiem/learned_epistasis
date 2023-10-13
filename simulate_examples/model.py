import torch
import torch.nn as nn
import torch.nn.functional as F

input_size1 = 85
output_size1 = 64
n_embd = 32
head_size = 16
num_heads = 4 #head_size must be divisible by num_heads

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("communication_matrix", torch.ones(input_size1,input_size1))

    def forward(self, x):
        # Input (batch, input_size, n_embd)
        # Output (batch, input_size, head_size)
        k = self.key(x) # (batch, input_size, head_size)
        q = self.query(x)  # (batch, input_size, head_size)
        W = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (batch, input_size, input_size)
        W = W.masked_fill(self.communication_matrix[:input_size1, :input_size1] == 0, float('-inf')) # (batch, input_size, input_size)
        W = F.softmax(W, dim=-1)

        v = self.value(x) # (batch, input_size, head_size)
        out = W @ v # (batch, input_size, head_size)
        return out
    
class MultiHead(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size*num_heads,n_embd)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads],dim=-1) #(batch,input_size,head_size (global))
        x = self.linear(x) #(batch,input_size,n_embd)
        return x
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x) #(batch,input_size,n_embd)
    
class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # input = output = (batch,input_size,n_embd)
        x = x + self.multihead(x) 
        x = x + self.ffwd(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(input_size1, n_embd) #could embed rank and file separately
        #self.head = Head(head_size)
        self.blocks = nn.Sequential(*[Block() for _ in range(3)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear = nn.Linear(input_size1*n_embd,output_size1)

    def forward(self, x):
        batch, input_size = x.shape
        x = F.one_hot(x, num_classes=n_embd) # (batch, input_size, n_embd)
        square_embd = self.pos_embedding(torch.arange(input_size)) # (input_size, n_embd)
        x = x + square_embd #position embedding #(batch, input_size, n_embd)
        x = self.blocks(x) #(batch, input_size, n_embd)
        x = x.view(batch, input_size1*n_embd)
        x = self.linear(x)
        # x = self.multihead(x) #batch, input_size, head_size
        # x = x.view(batch, input_size*head_size)
        # x = self.linear1(x) 
        # x = self.relu(x)
        # x = self.linear2(x)

        return x

    