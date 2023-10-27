from processing import *
from models import *
import multiprocessing
import torch
import torch.nn as nn

GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 10000
num_epochs = 200
batch_size = 64
train_prop = 0.9
num_estimate = 500
lr = 0.000001

X1 = [convert_sampling_file("sampled_genotypes/sample_" + str(i)) for i in range(num_files)]
X1 = torch.tensor(X1) - 1
GetMemory()
X1 = X1.repeat_interleave(2,dim=0)
print(X1.shape)

X2 = [convert_command_file1("../commands/command_" + str(i)) for i in range(num_files)]
X2 = torch.tensor(X2)
X2 = X2.reshape(num_files*2, -1)

GetMemory()

y = torch.tensor([i%2 for i in range(1,num_files*2 + 1)])

# Scramble data
ind = int(train_prop * X1.shape[0]) // 2
idx = torch.randperm(X1.shape[0] // 2)
idx = [2*i for i in idx[:ind]] + [2*i + 1 for i in idx[:ind]] + [2*i for i in idx[ind:]] + [2*i + 1 for i in idx[ind:]] #include both True and False example for each sample

X1 = X1[idx]
X2 = X2[idx]
y = y[idx]

# Split data
ind = int(train_prop * X1.shape[0])
X1_train, X1_test = X1[:ind], X1[ind:]
X2_train, X2_test = X2[:ind], X2[ind:]
y_train, y_test = y[:ind], y[ind:]

print(y_train.sum().item())
print(y_test.sum().item())


GetMemory()
GetTime()

# Define network
model = SimpleModel()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X1, X2, y = (X1_train, X2_train, y_train) if split == "train" else (X1_test, X2_test, y_test)
    idx = torch.randperm(X1.shape[0])[:num_samples]
    X1 = X1[idx]
    X2 = X2[idx]
    y = y[idx]
    if GPU_available:
        return X1.to("cuda"), X2.to("cuda"), y.to("cuda")
    else:
        return X1, X2, y

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X1, X2, y = get_batch(split, num_samples)
        y_pred = torch.zeros(num_samples)
        for i in range(0, num_samples, batch_size):
            try:
                X1_batch = X1[i:i+batch_size]
                X2_batch = X2[i:i+batch_size]
                y_pred[i:i+batch_size] = model(X1_batch, X2_batch)
            except IndexError:
                X1_batch = X1[i:]
                X2_batch = X2[i:]
                y_pred[i:] = model(X1_batch, X2_batch)

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
estimate_loss(min(num_estimate, X1_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}.")
    GetTime()
    # Scramble data
    idx = torch.randperm(X1_train.shape[0])
    X1_train = X1_train[idx]
    X2_train = X2_train[idx]
    y_train = y_train[idx]
    
    for ind in range(0,X1_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X1_batch = X1_train[ind:ind+batch_size]
            X2_batch = X2_train[ind:ind+batch_size]
            y_batch = y_train[ind:ind+batch_size]
        except IndexError:
            X1_batch = X1_train[ind:]
            X2_batch = X2_train[ind:]
            y_batch = y_train[ind:]

        if GPU_available:
            X1_batch = X1_batch.to("cuda")
            X2_batch = X2_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X1_batch, X2_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X1_batch, X2_batch)


        loss = criterion(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    estimate_loss(min(num_estimate, X1_test.shape[0]))