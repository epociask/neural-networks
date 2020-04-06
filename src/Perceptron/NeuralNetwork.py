from src.Perceptron.Neuron import Neuron


class NeuralNetwork:

    def __init__(self, weights: list, hiddenLayerCount: int, inputLayer: list, outputLayerLength: int):
        """

        :param weights:
        :param hiddenLayerCount:
        :param inputLayer:
        :param outputLayerLength:
        """
        self.weights = weights
        self.bias = 0
        self.hiddenLayerCount = hiddenLayerCount
        self.inputLayer = inputLayer
        self.outputLayerLength = outputLayerLength
        self.initialize()
        self.hiddenLayers = []
        self.outputLayer = []

    def initialize(self):
        """

        :return:
        """
        assert len(self.weights) == len(self.inputLayer)
        for hl in range(self.hiddenLayerCount):
            temp = []
            i = len(self.weights) - 1

            while i is not 0:
                temp[i] = Neuron(self.weights, self.bias)
                i-=1

            self.hiddenLayers[hl] = temp

        for val in range(self.outputLayerLength):
            self.outputLayer[val] = Neuron(self.weights, self.bias)


    #TODO finish
    #remember inputs of next layer are outputs of prior layer
    def feed_forward(self):

        for row in self.hiddenLayers:
