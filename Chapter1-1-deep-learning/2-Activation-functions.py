# An "activation function" is a function applied at each node. 
# It converts the node's input into some output.
# The rectified linear activation function (called ReLU) has been shown to lead to very high-performance networks. 
# This function takes a single number as an input, returning 0 if the input is negative, and the input if the input is positive.

def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Import package
import numpy as np

# The input data has been pre-loaded as input_data
input_data = np.array([3, 5])

# The weights are available in a dictionary called weights
weights = {'node_0': array([2, 4]), 'node_1': array([ 4, -5]), 'output': array([2, 7])}

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)