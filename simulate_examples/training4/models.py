import torch
import torch.nn as nn
import torch.nn.functional as F

num_chrom = 100
len_chrom = 1000

n_embd = len_chrom
input_size = num_chrom

dropout = 0.5


class PairwiseSimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.Linear(3*input_size*2, 4*input_size*2)
        self.ln2 = nn.Linear(4*input_size*2, input_size*2)
        self.ln3 = nn.Linear(input_size*2, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = F.one_hot(x).float()
        x = x.reshape(x.shape[0],-1)

        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.ln3(x)
        x = self.sigmoid(x)
        x = x.reshape(-1)

        return x