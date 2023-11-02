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

sigma = 4

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
    with open(file,"r") as f:
        lines, = f.readlines()

    point = float(lines.split()[-2])
    arange = np.arange(num_samples)

    ind = round(point*num_samples - 0.5)
    #out1[ind] = 1.0

    out1 = Normal(arange,ind)
    out1 = out1 / out1[ind] #Normalize peak to have value 1
    out1 = out1.tolist()

    while True:
        rand_ind = random.randint(0,num_samples-1)
        if abs(rand_ind - ind) > 20:
            break

    out2 = Normal(arange,rand_ind)
    out2 = out2 / out2[rand_ind]
    out2 = out2.tolist()

    return [out1,out2]

def convert_command_file2(command_file):
    with open(command_file, "r") as f:
        string, = f.readlines()

    points = [float(s) for s in string.split(" ")[6:8]]

    out1 = [0.0 for _ in range(num_samples)]
    out2 = [0.0 for _ in range(num_samples)]

    inds = [round(point*num_samples - 0.5) for point in points]



    out1[inds[0]] = 1
    out1[inds[1]] = 1

    while True:
        point_1 = random.randint(0,num_samples-1)
        point_2 = random.randint(0,num_samples-1)
        if point_1 != point_2 and [point_1,point_2] != inds and [point_2,point_1] != inds:
            break

    out2[point_1] = 1
    out2[point_2] = 1

    return out1, out2
