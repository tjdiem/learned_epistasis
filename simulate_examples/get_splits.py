import sys

selam_output = sys.argv[1]


# list of lists of split points
# Ancestry starts at 0 (a split point at 0 is inserted if chromosme starts at 1)
split_points = []



# Condense selam output into list of list of split points
with open(selam_output) as selam:
    lines = selam.readlines()

    chromosome = 0

    split_points.append([])

    for i in range(len(lines)):

        if lines[i][0] == '#':
            if lines[i-1][0] != "#":
                chromosome += 1
                split_points.append([])
            continue
        
        line = lines[i].split()

        if line[6] != '0' or line[7] != '0':
            split_points[chromosome].append(float(line[7]))


for i in range(len(split_points)):
    print()
    for split in split_points[i]:
        print(split)