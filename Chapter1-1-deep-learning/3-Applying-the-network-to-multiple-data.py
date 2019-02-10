#Define a function called predict_with_network() which will generate predictions for multiple data observations. 

# Define predict_with_network()
# that accepts two arguments - input_data_row and weights - and 
def predict_with_network(input_data_row, weights):
    # Calculate node 0 value
    # Calculate the input value of a node, multiply the relevant arrays together and compute their sum.
    node_0_input = (input_data_row * weights['node_0']).sum()
    # apply the relu() function
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)

def relu(input):
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Import package
import numpy as np

# The input data has been pre-loaded as input_data
input_data = [np.array([3, 5]), np.array([ 1, -1]), np.array([0, 0]), np.array([8, 4])]

# The weights are available in a dictionary called weights
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}

# Create empty list to store prediction results
results = []

for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)
        