from model import NeuralNetwork
from utils import load_csv, get_performance
from artist import rotate_number


def run():
    # Initialize size of network with

    # number of input, hidden and output nodes
    input_nodes = 784

    # Learning capacity
    hidden_nodes = 300

    # Number of possible answers
    output_nodes = 10

    # The delta to apply for error
    learning_rate = 0.1

    epochs = 5

    # Load data
    train_data = load_csv("data/mnist_train.csv")
    test_data = load_csv("data/mnist_test.csv")

    # Create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Training
    epochs = 5
    for e in range(epochs):
        print("Epoch: ", e)
        n.train_dataset(train_data)
        n.train_dataset([rotate_number(i) for i in train_data])
        n.train_dataset([rotate_number(i, -10) for i in train_data])

    # Testing - go through all the records in the test data set
    scorecard = n.test_dataset(test_data)

    print("performance =", get_performance(scorecard))

if __name__ == "__main__":
    run()
