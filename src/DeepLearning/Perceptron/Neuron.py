import math

e = math.e

sigmoid_activation = lambda val: 1 / (1 + e ** -val)

sigmoid_derivative = lambda val: (e ** -val) * sigmoid_activation(val) ** 2


class Neuron:
    """
    Simple neuron class
    """

    def __init__(self, bias: float):
        """
        Initializer
        :param weights: weights vertex
        :param bias: bias value
        """
        self.bias = bias
        self.activationValue = None
        self.sum_value = None

    def feed_forward(self, inputs: list, weights: list):
        """
        Feed forward function
        :param weights:
        :param inputs: input vertex
        :return: activation of dot product of weights & inputs + bias to receive activation value bounded between 0 & 1
        """
        temp = 0.0
        print("inputs passed to neuron : ", inputs)
        print("weights : ", weights)
        for index in range(len(inputs)):  # performs dot product w/ inputs and weights verticies
            temp += (inputs[index] * weights[index])
        self.sum_value = temp + self.bias
        self.activationValue = sigmoid_activation(self.sum_value)
        return self.activationValue


# weights = [0, 1]
# bias = 4
# n = Neuron(weights, bias)
#
# x = [2, 3]
# print(n.feed_forward(x))
