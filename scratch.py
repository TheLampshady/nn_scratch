from model import NeuralNetwork


def run():
    # Initialize size of network with
    inode = 3
    hnode = 4
    onode = 2

    n = NeuralNetwork(inode, hnode, onode)

    entry = [1.0, 0.5, -1.5]
    result = n.query(entry)

    print("Init:")
    print(n.w_input_hidden)

    target = list(result.T[0])
    n.train(entry, target)

    print("")
    print("Target:")
    print(n.w_input_hidden)

    target[0] += 1
    n.train(entry, target)

    print("")
    print("Error:")
    print(n.w_input_hidden)

if __name__ == "__main__":
    run()