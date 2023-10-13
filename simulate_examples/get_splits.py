import sys

id = ""
if len(sys.argv) > 1:
  id = sys.argv[1]

selam_output = "generated_files/selam_output"+id


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
            split_points[chromosome].append(line[7])

split_file = open('splits/split'+id, "w")

for i in range(len(split_points)):
    line = "\t".join(split_points[i])
    split_file.write(line + "\n")
