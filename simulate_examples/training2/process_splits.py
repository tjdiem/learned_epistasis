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




def convert_file(split_file, command_file):


   def convert_command_file(command_file, true_y):
       with open(command_file, "r") as f:
           string, = f.readlines()


       points = [float(s) for s in string.split(" ")[6:8]]


       out = [0 for _ in range(num_samples)]
       inds = [round(point*num_samples - 0.5) for point in points]


       if true_y:
           out[inds[0]] = 1
           out[inds[1]] = 1


       else:
           while True:
               point_1 = random.randint(0,num_samples-1)
               point_2 = random.randint(0,num_samples-1)
               if point_1 != point_2 and [point_1,point_2] != inds and [point_2,point_1] != inds:
                   break


           out[point_1] = 1
           out[point_2] = 1


       return out + [1]


  


   def convert_line(line):
      
       if line[:2] == "0\t":
           line = line.split("\t") + ["1"]
           val = -1
       else:
           line = ["0"] + line.split("\t") + ["1"]
           val = 1


       out = [0 for _ in range(num_samples)]
       split_points = [float(l) for l in line]
       for i in range(len(split_points) - 1):
           start = round(split_points[i]*num_samples)
           end = round(split_points[i+1]*num_samples)
           out[start:end] = [val for _ in range(end - start)]
           val *= -1




       return out + [0]
  
   # a = [[random.random() for _ in range(1001)] for _ in range(101)]
   # b = [[random.random() for _ in range(1001)] for _ in range(101)]
   # a[52][563] = 1
   # b[52][563] = 0


   # return [a,b]
   with open(split_file, "r") as f:
       lines = f.readlines()


   out = [convert_line(line) for line in lines[:-1]]


   return [out + [convert_command_file(command_file,True)], out + [convert_command_file(command_file,False)]]
























