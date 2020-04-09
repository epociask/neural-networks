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
        self.tempDict = {}

    def toString(self) -> str:

        return f"InputLayer {self.inputLayer}\nHidden Layer {self.hiddenLayers} \nOutputLayer {self.outputLayer}"

    def initialize(self):
        """

        :return:
        """
        w = 0
        for hl in range(self.hiddenLayerCount):
            temp = []
            i = len(self.inputLayer[0]) - 1

            while i >= 0:
                print("iiiiiiiiiiiiiiiiii", i)
                temp.append(Neuron(self.bias))
                w += 1
                i -= 1

            self.hiddenLayers.append(temp)

        for val in range(self.outputLayerLength):
            self.outputLayer.append(Neuron(self.bias))

        # print("hidden layers", self.hiddenLayers)

    # TODO finish
    # remember inputs of next layer are outputs of prior layer
    def feed_forward(self):
        # print("Input layer ", self.inputLayer)

        weight_count = 0
        first = True
        index_ref = 0
        count = 0
        print(self.hiddenLayers)
        for column in self.hiddenLayers:
            print("Column", column)
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
        self.inputLayer = input_data
        for epoch in range(epochs):
            index = 0

            for row in input_data:
                clear()
                if not self.initialized:
                    print("Initializing")
                    self.initialize()
                    self.initialized = True

                self.inputLayer = row

                self.feed_forward()

                print(self.tempDict)
                new_weights = []
                # print("input row : ", row)
                print("index val", index)
                print(self.outputLayer)
                y_true = expected_output[index]

                print('---------------------------------------\n\t\tTRAINING \n---------------------------------------')
                # print("weights vertex ", self.weights)
                print(epoch)
                y_pred = self.outputLayer[0].activationValue
                print("y prediction value : ", y_pred)
                print("y actual : ", y_true)
                print("expected output", expected_output[index])
                mse = square_error(expected_output[index], y_pred)
                dL_dYpred = -2 * (y_true - y_pred)
                increment = len(self.tempDict) - 1
                sum_out = self.outputLayer[0].sum_value
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
                        print(f"tempdict @ {increment}", self.tempDict)
                        for neuron in self.tempDict[increment]:
                            print(neuron)
                            temp_vals.append(neuron * sigmoid_derivative(sum_out))

                            if bias:
                                bias_vals.insert(0, sigmoid_derivative(sum_out))

                        first1 = True
                        while increment2 % 2 != 0 or first1:
                            print("increment2 value", increment2)
                            if first1:
                                first1 = False
                            node_derivatives.append(sigmoid_derivative(self.weights[increment2]))
                            increment2 += 1
                    increment -= 1

                    for neuron in self.hiddenLayers[0]:
                        for input in self.inputLayer:
                            temp_vals.insert(0, input * sigmoid_derivative(neuron.activationValue))

                        bias_vals.insert(0, sigmoid_derivative(neuron.activationValue))
                end = 5
                count = 0
                node = 0

                # d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                # d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                # d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                # # Neuron h1
                # d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                # d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                # d_h1_d_b1 = deriv_sigmoid(sum_h1)
                #
                # # Neuron h2
                # d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                # d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                # d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # # Neuron h1
                # self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                # self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                # self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                #
                # # Neuron h2
                # self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                # self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                # self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                for count in range(len(self.weights)):
                    bias = True
                    print("tempvalues", temp_vals)
                    print("node derivatives", node_derivatives)
                    print("count", count)
                    for weight in temp_weights:
                        print("tw", weight)
                    print("end", end)
                    if count <= 5:
                        print(count)
                        print("node", node)

                        try:
                            self.weights[count] = learning_rate * dL_dYpred * node_derivatives[node] * temp_vals[count]

                        except Exception:
                            self.weights[count] = learning_rate * dL_dYpred * node_derivatives[node - 1] * temp_vals[
                                count]

                        if count % 2 == 0 and count != 0:
                            print("updating bias")

                            if len(self.hiddenLayers[0]) > node:
                                self.hiddenLayers[0][node].bias = learning_rate * dL_dYpred * bias_vals[node]
                                node += 1


                            else:
                                self.outputLayer[0].bias = learning_rate * dL_dYpred * bias_vals[node]

                            #
                            # # Neuron o1
                            # self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                            # self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                            # self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                index += 1
                print("Mean Square Error : ", mse)

        self.inputLayer = [0, 1]
        self.feed_forward()
        print(f"predicted output for {self.inputLayer}: {self.outputLayer[0].activationValue}")
        print("--------------")
        print("--------------")
        print("--------------")


nn = NeuralNetwork([0, 1, 2, 3, 4, 5], 1, 1)
nn.train(100, [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], [1, 0, 1, 0, 1], .5)
