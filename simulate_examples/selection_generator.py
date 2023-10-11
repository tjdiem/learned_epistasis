import sys
from random import random


id = ""
if len(sys.argv) > 1:
  id = sys.argv[1]




#Epistatic selection parameters
site_1_loc = 0.05
site_2_loc = 0.35

strength    = 0.01
recessive = True


#Admixed population parameters
prop        = 0.2
generations = 200
Ne          = 10000


#sampling parameters
count = 50







selection = open("generated_files/selection"+id, "w")
selection.write("D\tA\t0\t0\t"+str(site_1_loc)+"\t"+str(site_2_loc)+"\t")

a = str(1 - strength)

if (recessive):
  selection.write("1\t1\t1\t1\t1\t1\t"+a+"\t1\t1\n")
else:
  selection.write("1\t1\t1\t"+a+"\t"+a+"\t1\t"+a+"\t"+a+"\t1\n")








# generate output file
output_string = str(generations)+"\t0\t"+str(count)+"\t0\tgenerated_files/selam_output"+id+"\n"
output = open("generated_files/output"+id, "w")
output.write(output_string)
output.close()


#generate demography file
demo = open("generated_files/demography"+id, "w")
demo.write("pop1\tpop2\tsex\t0\t1\n")
demo.write("0\t0\tA\t"+str(Ne)+"\t"+str(Ne)+"\n")
demo.write("0\ta0\tA\t"+str(prop)+"\t0\n")
demo.write("0\ta1\tA\t"+str(1-prop)+"\t0\n")
demo.close()
