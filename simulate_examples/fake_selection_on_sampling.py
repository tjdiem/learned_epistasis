import sys
import random

#1: input split file
#2: selection file
#3: 
#4:


split_file = open(sys.argv[1]).readlines()
if len(split_file[-1]) < 2:
    del split_file[-1]


id = sys.argv[1].split('_')[-1]


indvs_out  = int(sys.argv[2])


el0 = float(sys.argv[3]) #epi loc 0
el1 = float(sys.argv[4]) #epi loc 1
es  = float(sys.argv[5]) #epi strength
rec = sys.argv[6] == '1' #reccesive?
rl  = float(sys.argv[7]) #regular loc
rs  = float(sys.argv[8]) #regular strength



esel = [1,1,1, 1,1,1, 1,1,1]

if rec:
    esel[6] = 1 - es
else:
    esel[3] = 1 - es
    esel[4] = 1 - es
    esel[6] = 1 - es
    esel[7] = 1 - es

rsel = [1, 1 - rs/2, 1 - rs]

print(esel)


fitnesses = []



for i in range(0, len(split_file), 2):

    A_0 = 0  #A is epi loc 0
    A_1 = 0
    B_0 = 0  #B is epi loc 1
    B_1 = 0
    C_0 = 0  #C is regular site
    C_1 = 0

    chrom_0 = split_file[i ].split()
    chrom_1 = split_file[i+1].split()

    last_0 = 0
    last_1 = 0

    anc_0 = 0
    anc_1 = 0

    chrom_0.append("1")
    chrom_1.append("1")

    for split in chrom_0:
        split = float(split)

        if last_0 <= el0 and el0 < split:
            A_0 = anc_0

        if last_0 <= el1 and el1 < split:
            B_0 = anc_0
        
        if last_0 <= rs and rs < split:
            C_0 = anc_0
        
        
        last_0 = split

        anc_0 = 1 - anc_0
    
    for split in chrom_1:
        split = float(split)

        if last_1 <= el0 and el0 < split:
            A_1 = anc_1

        if last_1 <= el1 and el1 < split:
            B_1 = anc_1
        
        if last_1 <= rs and rs < split:
            C_1 = anc_1
        
        last_1 = split
        
        anc_1 = 1 - anc_1

    this_fit = 1

    if A_0 == 0 and A_1 == 0 and B_0 == 0 and B_1 == 0:
        print('0')
        this_fit = esel[0]
    elif A_0 == 0 and A_1 == 0 and B_0 + B_1 == 1:
        print('1')
        this_fit = esel[1]
    elif A_0 == 0 and A_1 == 0 and B_0 == 1 and B_1 == 1:
        print('2')
        this_fit = esel[2]
    elif A_0 + A_1 == 1 and B_0 == 0 and B_1 == 0:
        print('3')
        this_fit = esel[3]
    elif A_0 + A_1 == 1 and B_0 + B_1 == 1:
        print('4')
        this_fit = esel[4]
    elif A_0 + A_1 == 1 and B_0 == 1 and B_1 == 1:
        print('5')
        this_fit = esel[5]
    elif A_0 == 1 and A_1 == 1 and B_0 == 0 and B_1 == 0:
        print('6')
        this_fit = esel[6]
    elif A_0 == 1 and A_1 == 1 and B_0 + B_1 == 1:
        print('7')
        this_fit = esel[7]
    elif A_0 == 1 and A_1 == 1 and B_0 == 1 and B_1 == 1:
        print('8')
        this_fit = esel[8]
    else:
        print("no")
    
    #print(this_fit)

    
    if C_0 == 0 and C_1 == 0:
        this_fit *= rsel[0]
    elif C_0 == 1 and C_1 == 1:
        this_fit *= rsel[2]
    else:
        this_fit *= rsel[1]
    
    fitnesses.append(this_fit)






def weighted_random_choice(weighted_list, n):
    total_weight = sum(weighted_list)
    selected_indices = []

    for _ in range(n):
        # Generate a random number between 0 and the total weight
        random_weight = random.uniform(0, total_weight)
        
        # Iterate through the list to find the index where the random weight falls
        cumulative_weight = 0
        for i, weight in enumerate(weighted_list):
            cumulative_weight += weight
            if random_weight <= cumulative_weight:
                selected_indices.append(i)
                total_weight -= weight  # Reduce the total weight for subsequent selections
                weighted_list[i] = 0  # Mark this item as used (weight set to 0)
                break

    return selected_indices






new_individuals = weighted_random_choice(fitnesses, indvs_out)





new_split_lines = []

for i in new_individuals:
    new_split_lines.append(split_file[i*2])
    new_split_lines.append(split_file[i*2 + 1])



for line in new_split_lines:
    print(line, end='')

