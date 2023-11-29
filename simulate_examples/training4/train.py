from processing import *
from models import *
import multiprocessing
import concurrent.futures
import torch
import torch.nn as nn
from ast import literal_eval

GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 87500
num_epochs = 200
batch_size = 256
train_prop = 0.9
num_estimate = 500
lr = 0.0001


# with open("smaller_epistatic.txt","r") as f:
#     idx = literal_eval(f.read())

idx = list(range(87500))
idx = torch.randperm(len(idx))[:num_files].tolist()

X = [convert_files("../../test_training/sampled_genotypes/sample_stronger_" + str(i), "../../test_training/commands/command_stronger_" + str(i)) for i in idx] 
X = [x for x in X if x is not None]

X = torch.tensor(X)
X = X.reshape(-1, sample_width*num_chrom * 2)

GetMemory()

y = torch.tensor([i%2 for i in range(1,num_files*2 + 1)])


# Scramble data
ind = int(train_prop * X.shape[0]) // 2
idx = torch.randperm(X.shape[0] // 2)
idx = [2*i for i in idx[:ind]] + [2*i + 1 for i in idx[:ind]] + [2*i for i in idx[ind:]] + [2*i + 1 for i in idx[ind:]] #include both True and False example for each sample

X = X[idx]
y = y[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]

GetMemory()
GetTime()

# Define network
model = PairwiseSimpleModel()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y = (X_train, y_train) if split == "train" else (X_test, y_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    if GPU_available:
        return X.to("cuda"), y.to("cuda")
    else:
        return X, y

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, y = get_batch(split, num_samples)
        y_pred = torch.zeros(num_samples)
        for i in range(0, num_samples, batch_size):
            try:
                X_batch = X[i:i+batch_size]
                y_pred[i:i+batch_size] = model(X_batch)
            except IndexError:
                X_batch = X[i:]
                y_pred[i:] = model(X_batch)

        if GPU_available:
            y_pred = y_pred.to("cuda")                       


        
        loss = criterion(y_pred, y.float())
        predictions = (y_pred >= 0.5).int()
        print(predictions.sum().item()/num_samples)
        num_correct = (predictions == y).sum().item()
        print(f"Loss {split}: {loss.item():0.5f}, Accuracy {split}: {num_correct/num_samples:0.4f}")

    model.train()


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}.")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Shuffle in sampling dimension
    X = X.reshape(-1,2*sample_width,num_chrom)
    idx = torch.randperm(num_chrom)
    X = X[:,:,idx]
    X = X.reshape(-1,sample_width*num_chrom*2)
    
    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch = y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch = y_train[ind:]

        if GPU_available:
            X_batch = X_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)


        loss = criterion(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    estimate_loss(min(num_estimate, X_test.shape[0]))