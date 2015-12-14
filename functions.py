## - - - - - - - - - - - - - - - - - - - - - - - - ##
# sigmoid activation function
def sigmoid(x):
  return 1 / (1 + exp(-x))


## - - - - - - - - - - - - - - - - - - - - - - - - ##
# function to average rows of an array by block number
def blockrows(DATA, N):
	nrows, ncols = DATA.shape
	result = reshape(DATA, [N,nrows/N*ncols], order='F')
	result = mean(result, axis=0)
	return reshape(result, [nrows/N, ncols], order='F')


## - - - - - - - - - - - - - - - - - - - - - - - - ##
# function to train the model
def train_network(model):

	# get info from model dictionary
	numitems, numfeatures = model['inputs'].shape
	numcategories = len(unique(model['categories']))
	numblocks, numinitials, learnrate, weightrange = model['parameters']
	numupdates = numblocks * numitems


	# scale targets to [0 1]
	targets = matrix(model['inputs'])
	targets[targets<0] = 0

	# Iterate across initializations
	accuracy = empty([numupdates, numinitials])
	for K in range(numinitials):

		# get network weights and presentation order
		wts = uniform(-weightrange, weightrange, [numfeatures+1,numfeatures,numcategories])
		sequence = [j for i in range(numblocks) for j in permutation(numitems)]

		for U in range(numupdates):
			update_input  = matrix(model['inputs'][sequence[U]])
			update_class  = model['categories'][sequence[U]]
			update_target = targets[sequence[U]]

			# add bias
			update_input = insert(update_input,0,1)

			# forward pass
			output = empty([numcategories,numfeatures])
			for C in range(numcategories):
				output[C,:] = dot(update_input, wts[:,:,C])
			output = sigmoid(output) # comment out for linear

			# get SSE and compute classification probability
			Inverse_SSE = 1 / sum(square(output - update_target),axis=1)
			accuracy[U,K] = Inverse_SSE[update_class] / sum(Inverse_SSE)

			# update weights
			delta = output[update_class,:] - update_target
			delta = learnrate * dot(update_input.T, delta)
			wts[:,:,update_class] -= delta
		

	# block accuracy into rows, and average across initializations
	return mean(blockrows(accuracy,numitems),axis=1)

