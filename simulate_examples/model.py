import torch
import torch.nn as nn
import torch.nn.functional as F

input_size1 = 85
output_size1 = 64
n_embd = 32
head_size = 16
num_heads = 4 #head_size must be divisible by num_heads
num_blocks = 3

num_chrom = 100
len_chrom = 85
n_embd = 3


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
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # input = output = (batch,input_size,n_embd)
        x = x + self.multihead(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Embedding(len_chrom, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])
        self.multihead = MultiHead(num_heads, head_size // num_heads)
        self.linear = nn.Linear(len_chrom*n_embd,len_chrom*n_embd) #can change the output size of this
        self.ln1 = nn.LayerNorm(len_chrom*n_embd)

        ##
        self.ln2 = nn.LayerNorm(num_chrom*len_chrom*n_embd)

        self.linear2_1 = nn.Linear(num_chrom*len_chrom*n_embd,100)
        self.linear3_1 = nn.Linear(100,1)

        self.linear2_2 = nn.Linear(num_chrom*len_chrom*n_embd,100)
        self.linear3_2 = nn.Linear(100,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # X (batch, num_chrom, len_chrom, n_embd)
        #print(x.shape)
        #pos_embd = self.pos_embedding(torch.arange(len_chrom)) # (len_chrom, n_embd)
        #x = x + pos_embd #(batch, num_chrom, len_chrom, n_embd)
        #x = torch.cat((x, torch.zeros(len_chrom)),dim=2)
        x = self.blocks(x) #(batch, num_chrom, len_chrom, n_embd)
        x = x.view(x.shape[0], num_chrom, len_chrom*n_embd) #(batch, num_chrom, len_chrom * n_embd)
        x = self.ln1(x) #(batch, num_chrom, len_chrom * n_embd)
        x = self.linear(x) #(batch, num_chrom, len_chrom * n_embd)

        x = x.view(x.shape[0], num_chrom*len_chrom*n_embd)
        x = self.ln2(x)

        y1 = self.linear2_1(x)
        y1 = self.relu(y1)
        y1 = self.linear3_1(y1)

        y2 = self.linear2_2(x)
        y2 = self.relu(y2)
        y2 = self.linear3_2(y2)

        # print(x.shape)
        # x = self.multihead(x) #batch, input_size, head_size
        # x = x.view(batch, input_size*head_size)
        # x = self.linear1(x) 
        # x = self.relu(x)
        # x = self.linear2(x)

        return y1, y2

    