import torch
from model import *

file = "splits/split"

fixed_chrom_len = 85

def convert_line(line):
    if line[:2] == "0\t":
        line = line.split("\t")
        a = 0
    else:
        line = ["0"] + line.split("\t")
        a = 1

    #lengths = [float(line[i+1]) - float(line[i]) for i in range(len(line) - 1)]
    #lengths = [float(line[i+1]) - float(line[i]) for i in range(min(len(line) - 1,fixed_window_length))] + [0 for _ in range(max(0,fixed_window_length - len(line) + 1))]

    diff = fixed_chrom_len - len(line) + 1

    if diff >= 0:
        split_points = [float(l) for l in line]
        starts = split_points[:-1] + [1 for _ in range(diff)]
        ends = split_points[1:] + [1 for _ in range(diff)]
        nums = [-1**i for i in range(a,len(split_points)+a-1)] + [0 for _ in range(diff)] #possibly don't need this but safe to include for now
    else:
        split_points = [float(l) for l in line[:fixed_chrom_len+1]]
        starts = split_points[:-1]
        ends = split_points[1:]
        nums = [-1**i for i in range(a,len(starts)+a)]

    return [starts,ends,nums]

with open(file, "r") as f:
    lines = f.readlines()


X = [convert_line(line) for line in lines[:-1]]

X = torch.tensor(X)
X = X.transpose(-2,-1)
print(X.shape)

X_train = torch.zeros(256,100,85,3)
X_train = X_train + X

model = Model()
model.train()
y1, y2 = model(X_train)
# print(y1,y2)

# X (100 samples x variable positions (will be normalized) x 2 (length and category))

#transformer model without positional encoding, since positions are already kind of already part of the input