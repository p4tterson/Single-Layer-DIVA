
## - - - - - - - - - - - - - - - - - - - - - - - - ##
# function to train the model
def train_network(model):

	# get info from model dictionary
	INPUTS, ASSIGNMENTS = model['inputs'], model['categories']
	NUMBLOCKS, NUMINITS, LR, WR = model['parameters']
	DIAGONALCONNS = model['diagonalconnections']

	# get various dimensions
	NUMINPUTS, NUMFEATURES = INPUTS.shape
	NUMCLASSES = len(np.unique(ASSIGNMENTS))

	# --- add bias unit to input patterns
	in_with_bias = insert(INPUTS,0,1,axis=1)

	# Iterate across initializations
	accuracy = empty([NUMBLOCKS, NUMINITS])
	for K in range(NUMINITS):

		# ----- get network weights
		wts = uniform(-WR, WR, [NUMFEATURES+1,NUMFEATURES,NUMCLASSES])

		# zero out diagonal connections if desired
		if not DIAGONALCONNS:
			# get index of same-dimension connections
			diagonal_index = np.eye(NUMFEATURES) == 1
			diagonal_index = insert(diagonal_index, 0, False, axis=0)

			# force weights to 0
			wts[diagonal_index] = 0


		# iterate across blocks
		for B in range(NUMBLOCKS):

			# run forward pass
			output = forwardpass(in_with_bias, wts)

			# get classification responses, store accuracy
			responses = responserule(output,INPUTS)

			# store mean accuracy
			accuracy[B,K] = mean(np.equal(ASSIGNMENTS,responses))

			# update weights
			wts = weightupdate(wts, output, INPUTS, ASSIGNMENTS, LR, DIAGONALCONNS)

	# block accuracy into rows, and average across initializations
	return mean(accuracy,axis=1)


# ------------------------------------------------------
# ------------------------------------------------------
# propagate weights through network
def forwardpass(INPUTS,WTS, bias=True):
	NUMINPUTS = INPUTS.shape[0]
	_ , NUMFEATURES, NUMCLASSES = WTS.shape

	# dot prod inputs via weights from each category channel
	output = empty([NUMINPUTS,NUMFEATURES,NUMCLASSES])
	for C in range(NUMCLASSES):
		output[:,:,C] = dot(INPUTS, WTS[:,:,C])

	return output


# ------------------------------------------------------
# ------------------------------------------------------
# compute classification responses from output
def responserule(OUTPUT,TARGETS):
	NUMINPUTS, _ , NUMCLASSES = OUTPUT.shape

	# compute error on each category's reconstruction
	MSE = matrix(empty((NUMINPUTS,NUMCLASSES)))
	for C in range(NUMCLASSES):
		MSE[:,C] = mean(np.square(OUTPUT[:,:,C] - TARGETS) ,axis=1)

	# classify items based on best reconstruction
	return np.argmin(MSE,axis=1).T



# ------------------------------------------------------
# ------------------------------------------------------
# update weights using delta rule
def weightupdate(WTS, OUTPUT, INPUTS, ASSIGNMENTS, LR, DIAGONALCONNS):
	_ , NUMFEATURES , NUMCLASSES = OUTPUT.shape
	IN_BIAS = insert(INPUTS,0,1,axis=1)

	# Update each class on its exemplars
	for C in range(NUMCLASSES):

		# get index of class items
		INDS = np.equal(ASSIGNMENTS, C) 

		# compute weight change
		delta = OUTPUT[INDS,:,C] - INPUTS[INDS,:]
		delta = LR * dot(IN_BIAS[INDS,:].T, delta)

		# apply weight change
		WTS[:,:,C] -= delta

	# zero out diagonal connections if desired
	if not DIAGONALCONNS:
		# get index of same-dimension connections
		diagonal_index = np.eye(NUMFEATURES) == 1
		diagonal_index = insert(diagonal_index, 0, False, axis=0)

		# force them to 0
		WTS[diagonal_index] = 0

	return WTS
	