from Neuron import Neuron, sigmoid_derivative
import subprocess, platform
square_error = lambda y_true, y_pred: ((y_true - y_pred) ** 2)


def clear():
    if platform.system() == "Windows":
        subprocess.Popen('cls', shell=True).communicate()

    else:
        print('\033c', end="")

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
                temp.append(Neuron(self.bias))
                w += 1
                i -= 1

            self.hiddenLayers.append(temp)

        for val in range(self.outputLayerLength):
            self.outputLayer.append(Neuron(self.bias))

    # TODO finish
    # remember inputs of next layer are outputs of prior layer
    def feed_forward(self):
        # print("Input layer ", self.inputLayer)
        self.tempDict = {}
        weight_count = 0
        first = True
        index_ref = 0
        count = 0
        for column in self.hiddenLayers:
            print("weights", self.weights)
            tempList = []
            for neuron in column:
                # print("neuron #", index_ref)
                if first:
                    print("First")
                    tempList.append(neuron.feed_forward(self.inputLayer, self.weights[weight_count: weight_count + 2]))

                else:
                    tempList.append(
                        neuron.feed_forward(self.tempDict[count - 1], self.weights[weight_count:weight_count + 2]))
                index_ref += 1
                weight_count += len(column)

            self.tempDict[count] = tempList
            first = False
            count += 1

            # print("weight_count value ", weight_count)
            # print(tempDict)

        for index in range(len(self.outputLayer)):
            # print("count val ", count)
            # print("weight count val", weight_count)
            self.output.append(self.outputLayer[index].feed_forward(self.tempDict[count - 1],
                                                                    self.weights[weight_count: weight_count + 2]))

        print("Output layer ", self.outputLayer)

    def train(self, epochs, input_data: list, expected_output: list, learning_rate: float):
        """

        :param expected_output:
        :param input_data:
        :param epochs: number of times you'd like algorithm trained
        :return:
        """

        assert type(input_data[0]) == list, "Incorrect input values for input data... Please make sure to input as " \
                                            "list "
        # assert type(expected_output[0]) == list
        for epoch in range(epochs):
            index = 0

            for row in input_data:
                clear()
                new_weights = []
                # print("input row : ", row)
                y_true = self.outputLayer[index]
                self.inputLayer = row
                if not self.initialized:
                    self.initialize()
                    self.initialized = True
                print('---------------------------------------\n\t\tTRAINING \n---------------------------------------')
                self.feed_forward()
                # print("weights vertex ", self.weights)
                print(epoch)
                y_pred = self.output[0]
                print("y prediction value : ", y_pred)
                mse = square_error(expected_output[index], y_pred)
                dL_dYpred = -2 * (y_true - y_pred)
                increment = len(self.tempDict)
                sum_out = self.output[0].sum_value
                dic = {}
                first = True
                temp_vals = []
                bias_vals = []
                temp_weights = reversed(self.weights)
                node_derivatives = []
                increment2 = 0
                while increment >= 0:
                    if first:
                        bias = True
                        for neuron in self.tempDict[increment]:
                            temp_vals.insert(0, neuron.activationValue * sigmoid_derivative(sum_out))

                            if bias:
                                bias_vals.insert(0, sigmoid_derivative(sum_out))

                        first1 = True
                        while increment2 % 2 != 0 or not first1:

                            if first1:
                                first1 = False
                            node_derivatives.append(sigmoid_derivative(self.weights[increment2]))
                            increment2+=1
                end = 4
                count = 0
                node = 0
                for count in range(len(self.weights)):
                    bias = True
                    if not count < end:
                        self.weights[count] = learning_rate * dL_dYpred * node_derivatives[node] * temp_vals[]
                        count+=1

                        self.hiddenLayers[node][count].bias = learning_rate * dL_dYpred * bias_vals[1]
                        if count % 2 == 0:
                            node+=1



                            # # Neuron h1
                            # self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                            # self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                            # self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                            #
                            # # Neuron h2
                            # self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                            # self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                            # self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                            #
                            # # Neuron o1
                            # self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                            # self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                            # self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3


                    increment-=1
                index+=1
            print("Mean Square Error : ", mse)
            print("--------------")
            print("--------------")
            print("--------------")



nn = NeuralNetwork([0, 1, 2, 3, 4, 5], 1, 1)
nn.train(20, [[-2, -1], [25, 6], [17, 4], [-15, -6]], [1, 0, 0, 1], 4)
