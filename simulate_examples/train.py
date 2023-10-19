from process_splits import *
from models import *
import multiprocessing
import torch
import torch.nn as nn


GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 100_000
num_epochs = 15
batch_size = 256
train_prop = 0.9
lr = 0.001


# def func(i):
#     return convert_split_file("splits/split_" + str(i))


# Get Data
X = [convert_split_file("splits/split_" + str(i)) for i in range(100_000)]
# with multiprocessing.Pool(processes=8) as pool:
#     X = pool.map(func, range(num_files))
X = torch.tensor(X)
X = X.transpose(-2,-1)


print(X.shape)
GetMemory()


y = [convert_command_file("commands/command_" + str(i)) for i in range(100_000)]
y = torch.tensor(y)


y, _ = torch.sort(y, dim=1)
y1, y2 = y[:,0], y[:,1]


# Scramble data
idx = torch.randperm(X.shape[0])
X = X[idx]
y1 = y1[idx]
y2 = y2[idx]


# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
y_train1, y_test1 = y1[:ind], y1[ind:]
y_train2, y_test2 = y2[:ind], y2[ind:]


GetMemory()
GetTime()


# Define network
model = TransformerModel1()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
   X, y1, y2 = (X_train, y_train1, y_train2) if split == "train" else (X_test, y_test1, y_test2)
   idx = torch.randperm(X.shape[0])[:num_samples]
   X = X[idx]
   y1 = y1[idx]
   y2 = y2[idx]
   if GPU_available:
       return X.to("cuda"), y1.to("cuda"), y2.to("cuda")
   else:
       return X, y1, y2


@torch.no_grad()
def estimate_loss(num_samples):
   model.eval()
   for split in ["train", "test"]:
       X, y1, y2 = get_batch(split, num_samples)
       y_pred1 = torch.zeros(num_samples)
       y_pred2 = torch.zeros(num_samples)
       for i in range(0, num_samples, batch_size):
           try:
               X_batch = X[i:i+batch_size]
               y_pred1[i:i+batch_size], y_pred2[i:i+batch_size] = model(X_batch)
           except IndexError:
               X_batch = X[i:]
               y_pred1[i:], y_pred2[i:] = model(X_batch)


       if GPU_available:
           y_pred1 = y_pred1.to("cuda")
           y_pred2 = y_pred2.to("cuda")
                               
          
       # try:
       #     y_pred1, y_pred2 = model(X)
       # except torch.cuda.OutOfMemoryError:
       #     torch.cuda.empty_cache()
       #     y_pred1, y_pred2 = model(X)
      
       print(y_pred1)
       avg_dist = (abs(y_pred1 - y1) + abs(y_pred2 - y2)).sum().item()/num_samples/2
       loss = criterion(y_pred1, y1) + criterion(y_pred2, y2)
       print(f"Loss {split}: {loss.item():0.5f}, Avg dist {split}: {avg_dist}")


   model.train()




if GPU_available:
   print("GPU is available.")
   model = model.to("cuda")
   criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
   print("No GPU available. Running on CPU.")


GetMemory()
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
   y_train1 = y_train1[idx]
   y_train2 = y_train2[idx]
  
   for ind in range(0,X_train.shape[0],batch_size):


       #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())


       try:
           X_batch = X_train[ind:ind+batch_size]
           y_batch1 = y_train1[ind:ind+batch_size]
           y_batch2 = y_train2[ind:ind+batch_size]
       except IndexError:
           X_batch = X_train[ind:]
           y_batch1 = y_train1[ind:]
           y_batch2 = y_train2[ind:]


       if GPU_available:
           X_batch = X_batch.to("cuda")
           y_batch1 = y_batch1.to("cuda")
           y_batch2 = y_batch2.to("cuda")


       try:
           y_pred1, y_pred2 = model(X_batch)
       except torch.cuda.OutOfMemoryError:
           torch.cuda.empty_cache()
           y_pred1, y_pred2 = model(X_batch)




       loss = criterion(y_pred1, y_batch1) + criterion(y_pred2, y_batch2)


       optimizer.zero_grad()
       loss.backward()
       optimizer.step()


   estimate_loss(2000)




"""
record:
guess 1/3 and 2/3 for every input: Loss test: 0.11056, Avg dist test: 0.1968602294921875
Simple 2 layer FFNN (SimpleModel): Loss test: 0.10467, Avg dist test: 0.19030361938476562
"""











