from src.Perceptron.Neuron import Neuron

mean_square_error = lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
    # TODO autogenerate  weights
    # TODO only should accept inputLayer and Output layer as parameters
    def __init__(self, weights: list, hiddenLayerCount: int, outputLayerLength: int):
        """

        :param weights: hyperparamter
        :param hiddenLayerCount:
        :param inputLayer:
        :param outputLayerLengthLength:
        """
        self.weights = weights
        self.bias = 0
        self.hiddenLayerCount = hiddenLayerCount
        self.inputLayer = []
        self.outputLayerLength = outputLayerLength
        self.outputLayer = []
        self.hiddenLayers = []
        self.output = []
        self.initialized = False

    def toString(self) -> str:

        return f"InputLayer {self.inputLayer}\nHidden Layer {self.hiddenLayers} \nOutputLayer {self.outputLayer}"

    def initialize(self):
        """

        :return:
        """
        w = 0
        for hl in range(self.hiddenLayerCount):
            temp = []
            i = len(self.inputLayer)

            while i is not 0:
                temp.append(Neuron(self.weights[0], self.bias))
                w += 1
                i -= 1

            self.hiddenLayers.append(temp)

        for val in range(self.outputLayerLength):
            self.outputLayer.append(Neuron(self.weights, self.bias))

    # TODO finish
    # remember inputs of next layer are outputs of prior layer
    def feed_forward(self):
        # print("Input layer ", self.inputLayer)
        tempDict = {}
        weight_count = 0
        first = True
        index_ref = 0
        count = 0
        for column in self.hiddenLayers:
            # print("column ", column)
            tempList = []
            for neuron in column:
                # print("neuron #", index_ref)
                if first:
                    tempList.append(neuron.feed_forward(self.inputLayer, self.weights[weight_count: weight_count + 2]))

                else:
                    tempList.append(
                        neuron.feed_forward(tempDict[count - 1], self.weights[weight_count:weight_count + 2]))
                index_ref += 1

            tempDict[count] = tempList
            first = False
            count += 1

            weight_count += len(column)
            # print("weight_count value ", weight_count)
            # print(tempDict)

        for index in range(len(self.outputLayer)):
            # print("count val ", count)
            # print("weight count val", weight_count)
            self.output.append(self.outputLayer[index].feed_forward(tempDict[count - 1],
                                                                    self.weights[weight_count: weight_count + 2]))

        print("Output layer ", self.outputLayer)

    def train(self, epochs, input_data: list, expected_output: list):
        """

        :param expected_output:
        :param input_data:
        :param epochs: number of times you'd like algorithm trained
        :return:
        """

        assert type(input_data[0]) == list
        assert type(expected_output[0]) == list
        for epoch in range(epochs):

            for row in input_data:
                print("input row : ", row)
                self.inputLayer = row

                if not self.initialized:
                    self.initialize()
                    self.initialized = True
                print('---------------------------------------\n\t\tFeeding forward\n-----------------------------')
                self.feed_forward()
                print("weights vertex ", self.weights)
                print(epoch)
                y_pred = self.output[0]
                print("y prediction value : ", y_pred)


nn = NeuralNetwork([0, 1, 1, 1, 1, 1], 1, 1)
nn.train(4, [[2, 4], [1, 3], [2, 8], [5, 5]], [[0], [0], [1], [1]])
