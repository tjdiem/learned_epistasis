import sys


id = sys.argv[1]

ancestries = []


chrom_size = 1 #TODO parameterize this
samples = 1000

window_size = chrom_size / samples


with open("genotypes/phase_"+id) as phase_file:
    phase_file = phase_file.readlines()


    for k in range(len(phase_file) // 2):
        splits = phase_file[k*2]
        phases = phase_file[k*2 + 1]

        if(len(phases) > 2):
            ancestries.append(['0']*samples)
        else:
            continue

        j = 0

        splits = splits.split()
        phases = phases.split()

        x = 0
        anc = '0'
                
        for h in range(len(splits)):
            split = splits[h]
                    
            split = float(split)
                    
            while(x < split):
                
                ancestries[-1][j] = anc

                j += 1
                x += window_size
                    
            anc = phases[h]






with open("sampled_genotypes/sample_"+id, 'w') as sample_file:
    for i in range(len(ancestries)):
        sample_file.write("".join(ancestries[i])+"\n")


