import numpy
# scipy.special for the sigmoid function expit()
import scipy.special

from utils import to_nn_input


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.3, normal=True):
        """
        Initializes the Neural Network with 3 layers, random weights for 2 matrices
            and sigmoid function
        :param input_nodes: int
        :param hidden_nodes: int
        :param output_nodes: int
        :param learning_rate: double
        """
        # Node Dimensions 
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Method for initializing weights
        self.normal = bool(normal)

        # Learning Rate
        self.lr = learning_rate
        
        # Activation Function (Sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        # Weights
        self.w_input_hidden = self._gen_weights(
            self.input_nodes, self.hidden_nodes
        )
        
        self.w_hidden_output = self._gen_weights(
            self.hidden_nodes, self.output_nodes
        )

    def query(self, inputs_list):
        """
        Processes the input throughout the neural network.
        :param inputs_list: list<double>: list side must equal self.input_nodes
        :return: list<double>: final outputs
        """
        # Convert to 2d array and transpose it
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.w_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.w_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def back_query(self, targets_list):
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.w_hidden_output.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hideen layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.w_input_hidden.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

    def train(self, inputs_list, targets_list):
        """

        :param inputs_list: list<double>: list side must equal self.input_nodes
        :param targets_list: list<double>: list side must equal self.output_nodes
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
        
        self.w_hidden_output += self._weight_delta(output_errors, hidden_outputs, final_outputs)
        self.w_input_hidden += self._weight_delta(hidden_errors, inputs, hidden_outputs)

    def train_dataset(self, train_data):
        for record in train_data:
            inputs = to_nn_input(record)
            # create the target output values
            # (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(self.output_nodes) + 0.01

            # all_values[0] is the target label for this record
            targets[int(record[0])] = 0.99
            self.train(inputs, targets)

    def test_dataset(self, test_data):
        scorecard = []
        for record in test_data:
            correct_label = int(record[0])
            inputs = to_nn_input(record)
            outputs = self.query(inputs)

            # the index of the highest value corresponds to the label
            label = numpy.argmax(outputs)
            # append correct or incorrect to list
            if label == correct_label:
                scorecard.append(1)
            else:
                scorecard.append(0)

        return scorecard

    def import_network(self, filename="export.nn"):
        pass

    def export_network(self, filename="export.nn"):
        pass

    def _gen_weights(self, nodes1, nodes2):
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

    def _weight_delta(self, errors, output1, output2):
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
