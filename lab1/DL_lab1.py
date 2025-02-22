import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1.0 - x)

def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 3)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(2, 2, 4)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def MSELoss(y_predict, y, calculate_Grad=True):
    if calculate_Grad:
        return (np.sum((y_predict - y) ** 2)) / y_predict.shape[0], 2 * (y_predict - y) / y_predict.shape[0]
    return (np.sum((y_predict - y) ** 2)) / y_predict.shape[0]

class Layer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size) / 100

    def forward(self, x):
        self.x = x
        self.z = sigmoid(np.dot(x, self.w) + self.b)
        return self.z

    def backward(self, pre_g, lr):
        self.b -= np.sum((pre_g * derivative_sigmoid(self.z))) * lr
        self.w -= np.dot(self.x.T, (pre_g * derivative_sigmoid(self.z))) * lr
        return np.dot((pre_g * derivative_sigmoid(self.z)), self.w.T)

class Model:
    def __init__(self, input_size = 2, hidden_size = 10, lr = 0.1):
        self.layer1 = Layer(input_size, hidden_size)
        self.layer2 = Layer(hidden_size, hidden_size)
        self.output = Layer(hidden_size, 1)
        self.lr = lr
        self.loss = []
        self.epoch = 0

    def train(self, x, y, epoch):
        self.epoch = epoch
        for i in range(epoch):
            output = self.output.forward(self.layer2.forward(self.layer1.forward(x)))
            loss, grad = MSELoss(output, y)
            self.layer1.backward(self.layer2.backward(self.output.backward(grad, self.lr), self.lr), self.lr)
            self.loss.append(loss)
            if i % 5000 == 0: #control how many cycle print loss
                print(f"epoch {i}  loss : {loss}")
        self.prediction = output
        plt.subplot(2, 1, 1)
        plt.title("Learning Curve", fontsize=18)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.loss)
        return output

    def show_result(self, x, y):
        print(f"Accuracy : {sum((self.prediction > 0.5) == (y == 1)) / y.size}")
        print("Prediction : ")
        for i in range(y.size):
            print(f"Iter{i + 1} |    Ground truth: {y[i]} |     prediction: {self.prediction[i]} |")
        show_result(x, y, self.prediction > 0.5)

plt.figure(figsize=(12, 12))

# use linear as data
x, y = generate_linear()
model = Model(hidden_size=10, lr=0.1)
model.train(x, y, 100000)
model.show_result(x, y)

# use XOR as data
#x, y = generate_XOR_easy()
#model = Model(hidden_size=10, lr=0.1)
#model.train(x, y, 100000)
#model.show_result(x, y)
