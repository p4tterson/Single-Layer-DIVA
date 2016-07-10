
Single Layer DIVA
---
This is a short python script to run simulations with a single-layer DIVA network:

             A1  A2    B1  B2
    OUTPUTS   █  █      █  █
    WTS         ┼ ┼    ┼ ┼
    INPUTS         █  █
                  I1  I2

`run.py` contains some examples of how to use the code. Basically, the user enters a set of parameters, inputs, and class labels into a dictionary object. The dictionary is passed to a `train_network` function in `functions.py`, which trains the network (using batch updates) and returns block-by-block training accuracy.

The performance of this network is notable due to its ability to acquire nonlinearly separable classifications. Accordingly, `run.py` tests model performance on three NLS classifications.

#####Requirements
Users of this script will need NumPy. I use 1.9.2 and 1.11.0 without issue, but i have not tested this on other versions. I *think* this works with Python 2 and 3, but again i have not extensively tested anything.