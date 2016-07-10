## - - - - - - - - - - - - - - - - - - - - - - - - ##
# Libraries
import numpy as np

# custom functions
execfile('functions.py')

## - - - - - - - - - - - - - - - - - - - - - - - - ##
# store info in model dictionaries

# a set of global parameters ---
# 	does not change across classifications
parameters =[	
		10, 		# Number of training blocks
		1,			# Number of initializations
		0.1,		# Network learning rate
		0.0]		# Initial weight range

# Basic 2D XOR Problem
XOR_2D = { 
	'inputs': np.matrix([ 
				[-1, -1],
				[-1,  1],
				[ 1, -1],
				[ 1,  1]]),
	'categories': [0, 1, 1, 0],
	'parameters': parameters,
		}

# Medin & Schwanenflugel (1981, E4) NLS Problem
Medin_NLS = { 
	'inputs': np.matrix([ 
				[-1, -1,  1],
				[-1,  1, -1],
				[-1,  1,  1],
				[ 1, -1, -1],
				[ 1, -1,  1],
				[ 1,  1, -1]]),
	'categories': [0, 1, 0, 0, 1, 1],
	'parameters': parameters,
		}

# 3-Bit Parity Problem, or Type VI from Shepard et al. (1961)
Three_Bit_Parity = { 
	'inputs': np.matrix([ 
				[-1, -1, -1],
				[-1, -1,  1],
				[-1,  1, -1],
				[-1,  1,  1],
				[ 1, -1, -1],
				[ 1, -1,  1],
				[ 1,  1, -1],
				[ 1,  1,  1]]),
	'categories': [0, 1, 1, 0, 1, 0, 0, 1],
	'parameters': parameters,
		}

## - - - - - - - - - - - - - - - - - - - - - - - - ##
# run simulations and print results to console
XOR_Accuracy = train_network(XOR_2D)
Medin_NLS_Accuracy = train_network(Medin_NLS)
Three_Bit_Parity_Accuracy = train_network(Three_Bit_Parity)

print('''\nTRAINING ACCURACY BY BLOCK:\n
\t\t\tXOR \tNLS \t3-Bit''')
for i in range(parameters[0]):
	S = '\t' + str(i+1) + ''
	for j  in [XOR_Accuracy, Medin_NLS_Accuracy, Three_Bit_Parity_Accuracy]:
		S += '\t\t' + str(round(j[i],3))
	print(S)

print('\nDone.\n')