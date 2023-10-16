import torch

file = "splits/split"

def convert_line(line):
    if line[:2] == "0\t":
        line = line.split("\t")
        a = 0
    else:
        line = ["0"] + line.split("\t")
        a = 1
    lengths = [float(line[i+1]) - float(line[i]) for i in range(len(line) - 1)]
    nums = [i % 2 for i in range(a,len(lengths)+a)] #possibly don't need this but safe to include for now

    return [[length,num] for length,num in zip(lengths,nums)]

with open(file, "r") as f:
    lines = f.readlines()


X = [convert_line(line) for line in lines[:-1]]

# X (100 samples x variable positions (will be normalized) x 2 (length and category))

#transformer model without positional encoding, since positions are already kind of already part of the input