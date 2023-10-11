# learned_epistasis
Using machine learning techniques to detect epistatic selection in ancestry marked admixed chromosomes.


First pull the SELAM submodule in
	git submodule update --init --recursive

Then make SELAM
	./make_selam

Then you can go into simulate_examples to make some splits
	cd simulate_examples
	./make_splits

The above command will make a file in simulate_examples/splits called split
	Each line of this file is a chromosome, each float in the line is a point at which the ancestry swaps from one population to the other, starting at population 0







 
