
## - - - - - - - - - - - - - - - - - - - - - - - - ##
# function to train the model
def train_network(model):

	# get info from model dictionary
	INPUTS, ASSIGNMENTS = model['inputs'], model['categories']
	NUMBLOCKS, NUMINITS, LR, WR = model['parameters']

	# get various dimensions
	NUMINPUTS, NUMFEATURES = INPUTS.shape
	NUMCLASSES = len(np.unique(ASSIGNMENTS))

	# --- add bias unit to input patterns
	in_with_bias = np.insert(INPUTS,0,1,axis=1)

	# Iterate across initializations
	accuracy = np.empty([NUMBLOCKS, NUMINITS])
	for K in range(NUMINITS):

		# ----- get network weights
		wts = np.random.uniform(-WR, WR, [NUMFEATURES+1,NUMFEATURES,NUMCLASSES])

		# iterate across blocks
		for B in range(NUMBLOCKS):

			# run forward pass and get classification responses
			output = forwardpass(in_with_bias, wts)
			responses = responserule(output,INPUTS)
			accuracy[B,K] = np.mean(np.equal(ASSIGNMENTS,responses))

			# update weights
			wts = weightupdate(wts, output, INPUTS, ASSIGNMENTS, LR)

	# block accuracy into rows, and average across initializations
	return np.mean(accuracy,axis=1)


# ------------------------------------------------------
# ------------------------------------------------------
# propagate weights through network
def forwardpass(INPUTS,WTS):
	NUMINPUTS = INPUTS.shape[0]
	_ , NUMFEATURES, NUMCLASSES = WTS.shape

	# inputs * weights for each category channel
	output = np.empty([NUMINPUTS,NUMFEATURES,NUMCLASSES])
	for C in range(NUMCLASSES):
		output[:,:,C] = np.dot(INPUTS, WTS[:,:,C])

	#  if logistic outs are desired, uncomment this line:
	# output = sigmoid(output)
	return output

# a quick logistic activation function
def sigmoid(z):
	return 1 / (1 + np.exp(-z))


# ------------------------------------------------------
# ------------------------------------------------------
# compute classification responses from output
def responserule(OUTPUT,TARGETS):
	NUMINPUTS, _ , NUMCLASSES = OUTPUT.shape

	# compute error on each category's reconstruction
	MSE = np.matrix(np.empty((NUMINPUTS,NUMCLASSES)))
	for C in range(NUMCLASSES):
		MSE[:,C] = np.mean(np.square(OUTPUT[:,:,C] - TARGETS) ,axis=1)

	# classify items based on best reconstruction
	return np.argmin(MSE,axis=1).T



# ------------------------------------------------------
# ------------------------------------------------------
# update weights using delta rule
def weightupdate(WTS, OUTPUT, INPUTS, ASSIGNMENTS, LR):
	_ , NUMFEATURES , NUMCLASSES = OUTPUT.shape
	IN_BIAS = np.insert(INPUTS,0,1,axis=1)

	# Update each class on its exemplars
	for C in range(NUMCLASSES):

		# get index of class items
		INDS = np.equal(ASSIGNMENTS, C) 

		# compute and apply weight change
		delta = OUTPUT[INDS,:,C] - INPUTS[INDS,:]
		delta = LR * np.dot(IN_BIAS[INDS,:].T, delta)
		WTS[:,:,C] -= delta

	return WTS
	