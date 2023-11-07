from processing import *
from models import *
import multiprocessing
import torch
import torch.nn as nn

GPU_available = torch.cuda.is_available()
print(GPU_available)


num_files = 50000
num_epochs = 500
batch_size = 64
train_prop = 0.9
num_estimate = 500
lr = 0.000001 * 1000


#Accuracy train: 0.8520
#Accuracy test: 0.7860
#
#num_files = 50000
#num_epochs = 200
#batch_size = 64
#train_prop = 0.9
#num_estimate = 500
#lr = 0.000001 * 1

saving = False

if saving:
    X = [create_input("../../data/sampled_genotypes/sample_stronger_" + str(i), "../../data/commands/command_stronger_" + str(i)) for i in range(num_files)]
    X = torch.tensor(X).float()
    torch.save(X, "X50k")
else:
    X = torch.load("X50k")




X = X.reshape(num_files*3, 6, num_chrom, piece_size)


GetMemory()



y = torch.tensor([((i)%3)%2 for i in range(1,num_files*3 + 1)])
I = torch.tensor([i % 3 for i in range(X.shape[0])])

print(y)




# Scramble data
ind = int(train_prop * X.shape[0]) // 3
idx = torch.randperm(X.shape[0] // 3)
idx = [3*i for i in idx[:ind]] + [3*i + 1 for i in idx[:ind]] + [3*i + 2 for i in idx[:ind]] +  [3*i for i in idx[ind:]] + [3*i + 1 for i in idx[ind:]] + [3*i + 2 for i in idx[ind:]]


X = X[idx]
y = y[idx]
I = I[idx]




# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train, y_test = y[:ind], y[ind:]
I_train, I_test = I[:ind], I[ind:]



GetMemory()
GetTime()


# Define network
model = EpiModel()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)







# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, y , I = (X_train, y_train, I_train) if split == "train" else (X_test, y_test, I_test)
    
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    y = y[idx]
    I = I[idx]

    if GPU_available:
        return X.to("cuda"), y.to("cuda"), I.to("cuda")
    else:
        return X, y, I

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:

        X, y , I = get_batch(split, num_samples)

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

        print("\t0\t1")
        for i in range(3):
            called_0 = torch.sum((predictions == 0) & (I == i)).item()
            called_1 = torch.sum((predictions == 1) & (I == i)).item()
            sums = called_0 + called_1
            called_0 /= sums
            called_1 /= sums
            print(i, f"{called_0:.3f}" , f"{called_1:.3f}", sep='\t')

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
    y_train  =  y_train[idx]
    I_train = I_train[idx]
    
    for ind in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())

        try:
            X_batch = X_train[ind:ind+batch_size]
            y_batch  =  y_train[ind:ind+batch_size]
        except IndexError:
            X_batch = X_train[ind:]
            y_batch  =  y_train[ind:]

        if GPU_available:
            X_batch  =  X_batch.to("cuda")
            y_batch  =  y_batch.to("cuda")

        try:
            y_pred = model(X_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch)


        loss = criterion(y_pred, y_batch.float())   # get loss

        optimizer.zero_grad()   #
        loss.backward()         #
        optimizer.step()        #loss.backward knows where to go

    estimate_loss(min(num_estimate, X_test.shape[0]))
