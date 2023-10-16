import sys


split_file = open(sys.argv[1]).readlines()

window = float(sys.argv[2])

for line in split_file:
    line = line.split()
    x = 0
    anc = True
    
    for split in line:
        anc = not anc
        split = float(split)
        while(x < split):
            if anc:
                print('@',end='')
            else:
                print('-',end='')
            x += window
    print()


for i in range(7):
    x = 0
    while x < 1:
        if len(str(x)) > i:
            print(str(x)[i], end='')
        x += window
        
    print()
