import models
import os
import subprocess
import time


fixed_chrom_len = models.len_chrom
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


def convert_command_file(file):
   with open(file, "r") as f:
       string, = f.readlines()


   return [float(s) for s in string.split(" ")[6:8]]


def convert_split_file(file):


   def convert_line(line):
       if line[:2] == "0\t":
           line = line.split("\t")
           a = 0
       else:
           line = ["0"] + line.split("\t")
           a = 1


       #lengths = [float(line[i+1]) - float(line[i]) for i in range(len(line) - 1)]
       #lengths = [float(line[i+1]) - float(line[i]) for i in range(min(len(line) - 1,fixed_window_length))] + [0 for _ in range(max(0,fixed_window_length - len(line) + 1))]


       diff = fixed_chrom_len - len(line) + 1


       if diff >= 0:
           split_points = [float(l) for l in line]
           starts = split_points[:-1] + [1 for _ in range(diff)]
           ends = split_points[1:] + [1 for _ in range(diff)]
           nums = [-1**i for i in range(a,len(split_points)+a-1)] + [0 for _ in range(diff)] #possibly don't need this but safe to include for now
       else:
           split_points = [float(l) for l in line[:fixed_chrom_len+1]]
           starts = split_points[:-1]
           ends = split_points[1:]
           nums = [-1**i for i in range(a,len(starts)+a)]


       return [starts,ends,nums]


   with open(file, "r") as f:
       lines = f.readlines()


   X = [convert_line(line) for line in lines[:-1]]
   return X





