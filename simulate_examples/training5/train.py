from processing import *
from models import *
import multiprocessing
import concurrent.futures
import torch
import torch.nn as nn
from ast import literal_eval
import sys

save_file = sys.argv[1] if sys.argv[1].endswith(".pth") else sys.argv[1] + ".pth"
data_dir = "../Data3"
GPU_available = torch.cuda.is_available()
print(GPU_available)
num_files = 1_000
num_epochs = 30
batch_size = 128
train_prop = 0.9
num_estimate = 5000
lr_start = 3e-4
lr_end = lr_start/100

GetMemory()

random.seed(random_seed)
X = [convert_files(data_dir + "/sampled_genotypes/sample_" + str(i), data_dir + "/commands/command_" + str(i)) for i in range(num_files)] 

random.seed(random_seed)
XI = [get_file_info(data_dir + "/commands/command_" + str(i)) for i in range(num_files) if X[i] is not None]

C = [convert_command_file1(data_dir + "/commands/command_" + str(i)) for i in range(num_files) if X[i] is not None]

X = [x for x in X if x is not None]


X = torch.tensor(X) - 1
X = X.reshape(-1, sample_width*num_chrom * 2)

XI = torch.tensor(XI)
XI = XI.reshape(X.shape[0], 4)

C = torch.tensor(C)
C = C.reshape(-1, 11)

GetMemory()

y = torch.tensor([i%2 for i in range(1,X.shape[0] + 1)])

# Scramble data
torch.manual_seed(random_seed)
ind = int(train_prop * X.shape[0]) // 2
idx = torch.randperm(X.shape[0] // 2)
idx = [2*i for i in idx[:ind]] + [2*i + 1 for i in idx[:ind]] + [2*i for i in idx[ind:]] + [2*i + 1 for i in idx[ind:]] #include both True and False example for each sample

X = X[idx]
XI = XI[idx]
y = y[idx]
C = C[idx]

# Split data
ind = int(train_prop * X.shape[0])
X_train, X_test = X[:ind], X[ind:]
XI_train, XI_test = XI[:ind], XI[ind:]
y_train, y_test = y[:ind], y[ind:]
C_train, C_test = C[:ind], C[ind:]

GetMemory()
GetTime()

# lr variables
lr = lr_start
lr_factor = (lr_end/lr_start)**(1/(num_epochs - 1))

# Define network
model = TransformerModel1()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Define functions for getting random batch and calculating loss
def get_batch(split, num_samples):
    X, XI, y, C = (X_train, XI_train, y_train, C_train) if split == "train" else (X_test, XI_test, y_test, C_test)
    idx = torch.randperm(X.shape[0])[:num_samples]
    X = X[idx]
    XI = XI[idx]
    y = y[idx]
    C = C[idx]
    if GPU_available:
        return X.to("cuda"), XI.to("cuda"), y.to("cuda"), C.to("cuda")
    else:
        return X, XI, y, C

@torch.no_grad()
def estimate_loss(num_samples):
    model.eval()
    for split in ["train", "test"]:
        X, XI, y, C = get_batch(split, num_samples)
        y_pred = torch.zeros(num_samples)
        for istart in range(0, num_samples, batch_size):

            iend = min(istart + batch_size, num_samples)

            X_batch = X[istart:iend]
            XI_batch = XI[istart:iend]
            y_pred[istart:iend] = model(X_batch, XI_batch)

        if GPU_available:
            y_pred = y_pred.to("cuda")                       

        weak_epistasis = (C[:,7] < C[:,10])
        weak_count = weak_epistasis.sum().item()
        
        loss = criterion(y_pred, y.float())
        predictions = (y_pred >= 0.5).int()


        print()
        print(split, "full set:")

        num_correct = (predictions == y).sum().item()
        print(f"Loss {split}: {loss.item():0.5f}, Accuracy {split}: {num_correct/num_samples:0.4f}")

        print("\t0\t1")
        for i in range(2):
            called_0 = torch.sum((predictions == 0) & (y == i)).item()
            called_1 = torch.sum((predictions == 1) & (y == i)).item()
            sums = called_0 + called_1
            called_0 /= sums
            called_1 /= sums
            print(i, f"{called_0:.3f}" , f"{called_1:.3f}", sep='\t')
        

            
        called_epistasic = (predictions == 1)
        
        print()
        print(split, "M and T evaluation")

        for i in range(8):
            minim = 0.1 + i/10
            maxum = 0.2 + i/10

            subset_0 = ((C[:,1] >= minim) & (C[:,1] < maxum))

            for j in range(10):

                minim = 100 + j*50
                maxum = 150 + j*50

                subset_1 = ((C[:,2] >= minim) & (C[:,2] < maxum))

                subset = (subset_0 & subset_1 & called_epistasic)

                #Counting proportion of those called epistasis which are correct

                subset_count = subset.sum().item()
                num_correct = ((predictions == y) & subset).sum().item()

                if subset_count == 0:
                    print("0",end='')
                    continue
                
                ratio = num_correct / subset_count

                if ratio < 0.5:
                    print(".",end='')
                elif ratio < 0.66:
                    print("-",end='')
                elif ratio < 0.80:
                    print("+",end='')
                elif ratio < 0.95:
                    print("$",end='')
                else:
                    print("#",end='')
            
            print()


        print()

        print(split, "epistrength and normstrength evaluation")

        for i in range(9):
            minim = 0.01 + i * 0.01
            maxum = 0.02 + i * 0.01

            subset_0 = ((C[:,7] >= minim) & (C[:,7] < maxum))

            for j in range(20):

                minim = -0.05  + j * 0.005
                maxum = -0.045 + j * 0.005

                subset_1 = ((C[:,10] >= minim) & (C[:,10] < maxum))

                subset = (subset_0 & subset_1 & called_epistasic)

                subset_count = subset.sum().item()
                num_correct = ((predictions == y) & subset).sum().item()

                if subset_count == 0:
                    print("0",end='')
                    continue
                
                ratio = num_correct / subset_count

                if ratio < 0.5:
                    print(".",end='')
                elif ratio < 0.66:
                    print("-",end='')
                elif ratio < 0.80:
                    print("+",end='')
                elif ratio < 0.95:
                    print("$",end='')
                else:
                    print("#",end='')
            
            print()


        print()

    model.train()


if GPU_available:
    print("GPU is available.")
    model = model.to("cuda")
    criterion = criterion.to("cuda")
#    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"
else:
    print("No GPU available. Running on CPU.")

GetMemory()
# estimate_loss(min(num_estimate, X_test.shape[0]))
#Training loop

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb: SIZE"

model.train()
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print(f"Started training on epoch {epoch + 1} of {num_epochs}, learning rate {lr:0.7f}")
    GetTime()
    # Scramble data
    idx = torch.randperm(X_train.shape[0])
    X_train = X_train[idx]
    XI_train = XI_train[idx]
    y_train = y_train[idx]

    # Shuffle in sampling dimension
    X_train = X_train.reshape(-1,2*sample_width,num_chrom)
    idx = torch.randperm(num_chrom)
    X_train = X_train[:,:,idx]

    # # Randomly switch two sites 
    # mask = torch.rand(X_train.shape[0]) < 0.5
    # shift_indices = torch.fmod(torch.arange(2*sample_width) + sample_width, 2*sample_width)
    # X_train[mask] = X_train[mask][:, shift_indices]
    # XI_train[mask][:,[-1,-2]] = XI_train[mask][:,[-2,-1]]

    X_train = X_train.reshape(-1,sample_width*num_chrom*2)

    # # Randomly muliply each example by 1 or -1
    # rand = torch.randint(0,2,size=(X_train.shape[0],)) * 2 - 1
    # XI_train[rand == -1][:,0] = 1 - XI_train[rand == -1][:,0]
    # rand = rand.unsqueeze(1)
    # X_train *= rand

    for istart in range(0,X_train.shape[0],batch_size):

        #print(torch.cuda.max_memory_allocated(),"and", torch.cuda.memory_allocated())
        iend = min(istart + batch_size, X_train.shape[0])
        
        X_batch = X_train[istart:iend]
        XI_batch = XI_train[istart:iend]
        y_batch = y_train[istart:iend]

        if GPU_available:
            X_batch = X_batch.to("cuda")
            XI_batch = XI_batch.to("cuda")
            y_batch = y_batch.to("cuda")

        try:
            y_pred = model(X_batch, XI_batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            y_pred = model(X_batch, XI_batch)


        loss = criterion(y_pred, y_batch.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr *= lr_factor
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    estimate_loss(min(num_estimate, X_test.shape[0]))
    # for child in model.children():
    #     if isinstance(child,nn.Linear):
    #         print("next")
    #         print("weight")
    #         print(child.weight)
    #         print("bias")
    #         print(child.bias)

if save_file.lower() != "none.pth":
    torch.save(model.state_dict(), save_file)
