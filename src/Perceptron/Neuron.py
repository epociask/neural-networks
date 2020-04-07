import math

e = math.e

sigmoid_activation = lambda val: 1 / (1 + e ** -val)

sigmoid_derivative = lambda val: (e ** -val) * sigmoid_activation(val) ** 2


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
        self.weight_track = 0

    def update_weights(self, weights: list):
        """
        Updates weights vertex w/ new values
        :param weights:
        :return: None
        """
        self.weights = weights

    def feed_forward(self, inputs: list, weights: list):
        """
        Feed forward function
        :param inputs: input vertex
        :return: activation of dot product of weights & inputs + bias to receive activation value bounded between 0 & 1
        """
        temp = 0.0
        print("inputs passed to neuron : ", inputs)
        print("weights : ", weights)
        for index in range(len(inputs)):  # performs dot product w/ inputs and weights verticies
            temp += (inputs[index] * weights[index])

        return sigmoid_activation(temp + self.bias)


# weights = [0, 1]
# bias = 4
# n = Neuron(weights, bias)
#
# x = [2, 3]
# print(n.feed_forward(x))
