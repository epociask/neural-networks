import math

e = math.e


def sigmoid_activation(x: float):
    """
    :param x:
    :return: activation value of x --> value in range(0.0, 1.0)
    """
    return 1 / (1 + e ** -x)


def sigmoid_derivative(x: float):
    """
    :param x:
    :return: d/dx activation value of x
    """
    return (e ** -x) * sigmoid_activation(x) ** 2


class Neuron:
    """
    Simple neuron class
    """
    def __init__(self, weights: list, bias: float):
        """
        Initializer
        :param weights: weights vertex
        :param bias: bias value
        """
        self.weights = weights
        self.bias = bias

    def update_weights(self, weights: list):
        """
        Updates weights vertex w/ new values
        :param weights:
        :return: None
        """
        self.weights = weights

    def feed_forward(self, inputs: list):
        """
        Feed forward function
        :param inputs: input vertex
        :return: activation of dot product of weights & inputs + bias to receive activation value bounded between 0 & 1
        """
        temp = 0.0
        for index in range(len(self.weights)):  # performs dot product w/ inputs and weights verticies
            temp += (inputs[index] * weights[index])

        return sigmoid_activation(temp + self.bias)


weights = [0, 1]
bias = 4
n = Neuron(weights, bias)

x = [2, 3]
print(n.feed_forward(x))
