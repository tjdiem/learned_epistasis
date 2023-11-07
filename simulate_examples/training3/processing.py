import models
import os
import subprocess
import time
import random
from math import sqrt, e, pi
import numpy as np
import matplotlib.pyplot as plt

random.seed(1337)
num_samples = models.len_chrom
num_chrom = 100
start_time = time.time()

sigma = 2.5

def GetMemory():
    if os.name == 'posix':
        mem_info = subprocess.check_output(['free','-b']).decode().split()
        total_memory = int(mem_info[7]) - int(mem_info[8]) 
        total_memory *= 10**-9
        
    elif os.name == 'nt':
        mem_info = subprocess.check_output(['wmic','OS','get','FreePhysicalMemory']).decode().split()
        total_memory = int(mem_info[1]) * 1024 * 10**-9
        
    print(f"Available memory: {total_memory:0.2f} GB")
    
def GetTime():
    seconds = int(time.time() - start_time)
    hours = seconds // 3600
    seconds = seconds % 3600
    minutes = seconds // 60
    seconds = seconds % 60
    print(f"Total time elapsed: {hours}h {minutes}m {seconds}s")

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

def convert_sampling_file(file):
    
    with open(file, "r") as f:
        lines = f.readlines()

    out = [[float(l) for l in line[:-1]] for line in lines]

    #random.shuffle(out) #This will shuffle the rows

    return out

def convert_command_file1(file):
    # convert selected site with normal distribution - standard deviation sigma can be adjusted
    with open(file,"r") as f:
        lines, = f.readlines()

    point = float(lines.split()[-2])
    arange = np.arange(num_samples)

    ind = round(point*num_samples - 0.5)

    out1 = Normal(arange,ind)
    out1 = out1 / out1[ind]
    out1 = out1.tolist()

    while True:
        rand_ind = random.randint(0,num_samples-1)
        if abs(rand_ind - ind) > 20:
            break

    out2 = Normal(arange,rand_ind)
    out2 = out2 / out2[rand_ind]
    out2 = out2.tolist()

    return [out1,out2]


def convert_command_file3(file):
    # convert selected site with "normal-ish" distribution

    with open(file,"r") as f:
        lines, = f.readlines()

    point = float(lines.split()[-2])
    out1 = [0.0 for _ in range(num_samples)]
    out2 = [0.0 for _ in range(num_samples)]
    ind = round(point*num_samples - 0.5)
    #out1[ind] = 1.0

    vals = [0.05, 0.1,0.15, 0.25,0.37, 0.52, 0.67, 0.77, 0.87, 0.95,0.87,0.77,0.67,0.52,0.37,0.25,0.15,0.1,0.051]
    d = len(vals) - 1
    inds = list(range(ind - d //2, ind + d//2 + 1))
    for val, ind in zip(vals,inds):
        if 0 <= ind < len(out1):
            out1[ind] = val

    

    while True:
        rand_ind = random.randint(0,num_samples-1)
        if abs(rand_ind - ind) > 20:
            break

    inds = list(range(rand_ind - d //2, rand_ind + d//2 + 1))
    for val, ind in zip(vals,inds):
        if 0 <= ind < len(out1):
            out2[ind] = val

    return [out1,out2]

def convert_command_file2(command_file):
    #convert epistatic sites with sum of normal distributions
    
    with open(command_file, "r") as f:
        string, = f.readlines()

    points = [float(s) for s in string.split(" ")[6:8]]

    inds = [round(point*num_samples - 0.5) for point in points]

    ind1, ind2 = sorted(inds)

    arange = np.arange(num_samples)

    out1 = Normal(arange,ind1) + Normal(arange,ind2)
    out1 = out1 / out1[ind1]
    out1 = out1.tolist()

    # out1[inds[0]] = 1
    # out1[inds[1]] = 1

    while True:
        rand_ind1 = random.randint(0,num_samples-1)
        rand_ind2 = random.randint(0,num_samples-1)
        rand_ind1, rand_ind2 = sorted([rand_ind1, rand_ind2])
        if abs(rand_ind1 - rand_ind2) > 10 and (abs(rand_ind1 - ind1) > 10 or abs(rand_ind2 - ind2) > 10):
            break

    out2 = Normal(arange,rand_ind1) + Normal(arange,rand_ind2)
    out2 = out2 / out2[rand_ind1]
    out2 = out2.tolist()
    

    # out2[rand_ind1] = 1
    # out2[rand_ind2] = 1

    return [out1, out2]

def convert_command_file4(command_file):
    # same as convert_command_file2 except False example will take one true epistatic site
    # this gets around 75% accuracy with simple model

    with open(command_file, "r") as f:
        string, = f.readlines()

    points = [float(s) for s in string.split(" ")[6:8]]

    ind1, ind2 = [round(point*num_samples - 0.5) for point in points]

    arange = np.arange(num_samples)

    out1 = Normal(arange,ind1) + Normal(arange,ind2)
    out1 = out1 / out1[ind1]
    out1 = out1.tolist()

    while True:
        rand_ind = random.randint(0,num_samples-1)
        if abs(rand_ind - ind1) > 15 and abs(rand_ind - ind2) > 15:
            break

    if random.random() < 0.5:
        out2 = Normal(arange,ind1) + Normal(arange,rand_ind)
    else:
        out2 = Normal(arange,ind2) + Normal(arange,rand_ind)

    out2 = out2 / out2[rand_ind]
    out2 = out2.tolist()

    return [out1, out2]

def convert_command_file5(command_file):
    # same as convert_command_file4 except False example will take selected site 
    # as one of the sites half the time

    with open(command_file, "r") as f:
        string, = f.readlines()

    points_str = string.split()
    points = [float(s) for s in points_str[6:8]]

    inds = [round(point*num_samples - 0.5) for point in points]

    ind1, ind2 = sorted(inds)

    arange = np.arange(num_samples)

    out1 = Normal(arange,ind1) + Normal(arange,ind2)
    out1 = out1 / out1[ind1]
    out1 = out1.tolist()

    if random.random() < 0.5:

        while True:
            false_ind = random.randint(0,num_samples-1)
            if abs(false_ind - ind1) > 15 and abs(false_ind - ind2) > 15:
                break

    else:
        point = float(points_str[-2])
        false_ind = round(point*num_samples - 0.5)

    if random.random() < 0.5:
        out2 = Normal(arange,ind1) + Normal(arange,false_ind)
    else:
        out2 = Normal(arange,ind2) + Normal(arange,false_ind)

    out2 = out2 / out2[false_ind]
    out2 = out2.tolist()

    return [out1, out2]
