import models
import os
import subprocess
import time
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(1337)
num_samples = models.len_chrom
num_chrom = 100
start_time = time.time()

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


def convert_sampling_file(file):
    
    with open(file, "r") as f:
        lines = f.readlines()

    out = [[float(l) for l in line[:-1]] for line in lines]

    #random.shuffle(out) #This will shuffle the rows

    return out














def convert_command_file1(file):
    with open(file,"r") as f:
        lines, = f.readlines()


    site1 = float(lines.split()[6])
    site2 = float(lines.split()[7])
    site3 = float(lines.split()[-2])

    out1 = [0.0 for _ in range(num_samples)]
    out2 = [0.0 for _ in range(num_samples)]
    out3 = [0.0 for _ in range(num_samples)]
    
    ind1 = round(site1*num_samples - 0.5)
    ind2 = round(site2*num_samples - 0.5)
    ind3 = round(site3*num_samples - 0.5)

    #out1[ind] = 1.0

    vals = [0.05, 0.1,0.15, 0.25,0.37, 0.52, 0.67, 0.77, 0.87, 0.95,0.87,0.77,0.67,0.52,0.37,0.25,0.15,0.1,0.051]
    d = len(vals) - 1
    
    inds = list(range(ind1 - d //2, ind1 + d//2 + 1))
    for val, ind in zip(vals,inds):
        if 0 <= ind < len(out1):
            out1[ind] = val

    inds = list(range(ind2 - d //2, ind2 + d//2 + 1))
    for val, ind in zip(vals,inds):
        if 0 <= ind < len(out2):
            out2[ind] = val
    
    inds = list(range(ind3 - d //2, ind3 + d//2 + 1))
    for val, ind in zip(vals,inds):
        if 0 <= ind < len(out3):
            out3[ind] = val

    return [[out1,out2] , [out2,out3] , [out3,out1]]