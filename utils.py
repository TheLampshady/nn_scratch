import csv
import numpy
import scipy.misc


def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)
    return data_list


def to_nn_input(record):
    return (numpy.asfarray(record[1:]) / 255.0 * 0.99) + 0.01


def open_my_number(filename):
    img_array = scipy.misc.imread(filename, flatten=True)
    img_data = 255.0 - img_array.reshape(784)
    return (img_data/ 255.0 * 0.99) + 0.01


def get_default_network(nn_class):
    input_nodes = 784
    hidden_nodes = 300
    output_nodes = 10
    learning_rate = 0.2

    # Load data
    train_data = load_csv("data/mnist_train.csv")

    # Create instance of neural network
    n = nn_class(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Training
    n.train_dataset(train_data)
    return n


def get_performance(results):
    scorecard_array = numpy.asarray(results)
    return scorecard_array.sum() / scorecard_array.size
