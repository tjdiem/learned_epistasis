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
piece_size = models.piece_size


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








def create_input(sample_file, command_file):
    with open(sample_file, "r") as f:
        lines = f.readlines()

    out = [[float(l) for l in line[:-1]] for line in lines]

    with open(command_file,"r") as f:
        lines, = f.readlines()
    

    site1 = float(lines.split()[6])
    site2 = float(lines.split()[7])
    site3 = float(lines.split()[-2])

    ind1 = int(round(site1*num_samples - 0.5))
    ind2 = int(round(site2*num_samples - 0.5))
    ind3 = int(round(site3*num_samples - 0.5))


    local_1 = []
    local_2 = []
    local_3 = []

    for i in range(num_chrom):
        piece_1 = [1] * piece_size
        piece_2 = [1] * piece_size
        piece_3 = [1] * piece_size

        for j in range(piece_size):
            index_1 = (j - piece_size//2) + ind1

            if index_1 >= 0 and index_1 < models.len_chrom:
                piece_1[j] = out[i][index_1]
            
            index_2 = (j - piece_size//2) + ind2
            if index_2 >= 0 and index_2 < models.len_chrom:
                piece_2[j] = out[i][index_2]
            
            index_3 = (j - piece_size//2) + ind3
            if index_3 >= 0 and index_3 < models.len_chrom:
                piece_3[j] = out[i][index_3]


        local_1.append(piece_1)
        local_2.append(piece_2)
        local_3.append(piece_3)
    
    pair_1 = [local_1,local_2] if (random.random() < 0.5) else [local_2,local_1]
    pair_2 = [local_2,local_3] if (random.random() < 0.5) else [local_3,local_2]
    pair_3 = [local_3,local_1] if (random.random() < 0.5) else [local_1,local_3]

    return [pair_1,pair_2,pair_3]






def convert_command_file(file):
    with open(file,"r") as f:
        lines, = f.readlines()


    site1 = float(lines.split()[6])
    site2 = float(lines.split()[7])
    site3 = float(lines.split()[-2])

    ind1 = round(site1*num_samples - 0.5)
    ind2 = round(site2*num_samples - 0.5)
    ind3 = round(site3*num_samples - 0.5)

    return [[ind1, ind2], [ind2, ind3], [ind3, ind1]]