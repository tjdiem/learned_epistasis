import sys
from random import random



id = sys.argv[1]


#Admixed population parameters
prop        = float(sys.argv[2])
generations = int(sys.argv[3])
Ne          = int(sys.argv[4])


#sampling parameters
count = int(sys.argv[5])

#Epistatic selection parameters
site_1_loc = float(sys.argv[6])
site_2_loc = float(sys.argv[7])

strength   = float(sys.argv[8])
recessive  = bool(sys.argv[9])      #True for recessive, False for dominant

#Regular selection site
site_3_loc       = float(sys.argv[10])
site_3_strength  = float(sys.argv[11])











selection = open("generated_files/selection_"+id, "w")

selection.write("D\tA\t0\t0\t"+str(site_1_loc)+"\t"+str(site_2_loc)+"\t")

a = str(1 - strength)

if (recessive):
  selection.write("1\t1\t1\t1\t1\t1\t"+a+"\t1\t1\n")
else:
  selection.write("1\t1\t1\t"+a+"\t"+a+"\t1\t"+a+"\t"+a+"\t1\n")

selection.write("S\tA\t0\t"+str(site_3_loc)+"\t1\t"+str(1 - site_3_strength/2)+"\t"+str(1 - site_3_strength)+"\n")





# generate output file
output_string = str(generations)+"\t0\t"+str(count)+"\t0\tgenerated_files/selam_output_"+id+"\n"
output = open("generated_files/output_"+id, "w")
output.write(output_string)
output.close()


#generate demography file
demo = open("generated_files/demography_"+id, "w")
demo.write("pop1\tpop2\tsex\t0\t1\n")
demo.write("0\t0\tA\t"+str(Ne)+"\t"+str(Ne)+"\n")
demo.write("0\ta0\tA\t"+str(prop)+"\t0\n")
demo.write("0\ta1\tA\t"+str(1-prop)+"\t0\n")
demo.close()
