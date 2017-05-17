import numpy
import scipy.special


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_notes, learning_rate=0.3, normal=True):
        """
        Initializes the Neural Network with 3 layers, random weights for 2 matrices
            and sigmoid function
        :param input_nodes: int
        :param hidden_nodes: int
        :param output_notes: int
        :param learning_rate: double
        """
        # Node Dimensions 
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_notes = output_notes

        # Method for initializing weights
        self.normal = bool(normal)

        # Learning Rate
        self.lr = learning_rate
        
        # Activation Function (Sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # Weights
        self.w_input_hidden = self.gen_weights(
            self.input_nodes, self.hidden_nodes
        )
        
        self.w_hidden_output = self.gen_weights(
            self.hidden_nodes, self.output_notes
        )

    def query(self, inputs_list):
        """
        Processes the input throughout the neural network.
        :param inputs_list: list<double>: list side must equal self.input_nodes
        :return: list<double>: final outputs
        """
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list):
        """

        :param inputs_list: list<double>: list side must equal self.input_nodes
        :param targets_list: list<double>: list side must equal self.output_notes
        :return:
        """
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.w_hidden_output.T, output_errors)
        
        self.w_hidden_output += self.weight_delta(output_errors, hidden_outputs, final_outputs)
        self.w_input_hidden += self.weight_delta(hidden_errors, inputs, hidden_outputs)
    
    def gen_weights(self, nodes1, nodes2):
        """
        Generates a matrix of size [n1 x n2] with random doubles for weights
        :param nodes1: int: size of nodes
        :param nodes2: int: size of nodes in next layer
        :return: matrix: Random
        """

        # Normal distribution of values with a standard deviation around 0.0
        if self.normal:
            return numpy.random.normal(
                0.0,
                pow(nodes2, -0.5),
                (nodes2, nodes1)
            )

        # More distributed set of values
        return numpy.random.rand(nodes2, nodes1) - 0.5

    def weight_delta(self, errors, output1, output2):
        """
        Distributes the change in error by size of weight and dot products
        :param errors:
        :param output1:
        :param output2:
        :return:
        """
        return self.lr * numpy.dot(
            errors * output2 * (1.0 - output2),
            numpy.transpose(output1)
        )
