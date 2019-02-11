# Forward propagation for a neural network with 2 hidden layers. Each hidden layer has two nodes. The input data has been preloaded as input_data. The nodes in the first hidden layer are called node_0_0 and node_0_1. Their weights are loaded as weights['node_0_0'] and weights['node_0_1'] respectively.
# The nodes in the second hidden layer are called node_1_0 and node_1_1. Their weights are loaded as weights['node_1_0'] and weights['node_1_1'] respectively.
# Create a model output from the hidden nodes using weights loaded as weights['output'].


def relu(input):
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

def predict_with_network(input_data):

    # Import package
    import numpy as np

    # The weights are available in a dictionary called weights
    weights = {'node_0_0': np.array([2, 4]),
    'node_0_1': np.array([ 4, -5]),
    'node_1_0': np.array([-1,  2]),
    'node_1_1': np.array([1, 2]),
    'output': np.array([2, 7])}

    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs *  weights['output']).sum()
    
    # Return model_output
    return(model_output)

# Import package
import numpy as np

# The input data has been pre-loaded as input_data
input_data = np.array([3, 5])

output = predict_with_network(input_data)
print(output)

