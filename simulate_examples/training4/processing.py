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

def convert_files(sampling_file, command_file):
    
    with open(sampling_file, "r") as f:
        lines = f.readlines()

    X = [[int(l) for l in line[:-1]] for line in lines]

    #random.shuffle(out) #This will shuffle the rows
    with open(command_file, "r") as f:
        s = f.readlines()[0].split()

    points = [float(s[6]), float(s[7]), float(s[10])]
    ind1, ind2, regular_ind = [round(num_samples*point - 0.5) for point in points]

    site1 = [sample[ind1] for sample in X]
    site2 = [sample[ind2] for sample in X]
    regular_site = [sample[regular_ind] for sample in X]

    out_true = site1 + site2 if random.random() < 0.5 else site2 + site1

    # rand = random.random()
    # if 0 <= rand < 0.25:
    #     out_false = site1 + regular_site
    # elif 0.25 <= rand < 0.5:
    #     out_false = site2 + regular_site
    # elif 0.5 <= rand < 0.75:
    #     out_false = regular_site + site1
    # else:
    #     out_false = regular_site + site2 

    out_false = site2 + regular_site if random.random() < 0.5 else regular_site + site2

    return [out_true, out_false]
