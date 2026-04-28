import numpy as np


class NeuralNetwork:                                        # FIX 1: Added indentation to entire class body

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 0.5):
        rng = np.random.default_rng(42)
        self.learning_rate = learning_rate
        self.w1 = rng.normal(scale=1.0, size=(input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(scale=1.0, size=(hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(output):
        return output * (1 - output)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def train(self, x, y, epochs: int = 10000):
        for epoch in range(epochs):
            output = self.forward(x)
            error = y - output
            loss = np.mean(np.square(error))

            d_output = error * self.sigmoid_derivative(output)
            d_hidden = np.dot(d_output, self.w2.T) * self.sigmoid_derivative(self.a1)

            self.w2 += np.dot(self.a1.T, d_output) * self.learning_rate
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate  # FIX 2: broken `.learning_rate` across lines — joined
            self.w1 += np.dot(x.T, d_hidden) * self.learning_rate
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate  # FIX 2: same fix applied here

            if epoch % 2000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

    def predict(self, x):
        return self.forward(x)


def main() -> None:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = NeuralNetwork(input_size=2, hidden_size=2, output_size=1,
                          learning_rate=0.5)
    model.train(x, y, epochs=10000)

    predictions = model.predict(x)
    print()
    print("Final predictions:")
    for sample, prediction in zip(x, predictions):
        print(f"Input: {sample.astype(int).tolist()} -> Output: {prediction[0]:.4f}")  # FIX 3: broken f-string `{prediction` across lines — joined


if __name__ == "__main__":
    main()                                                  # FIX 4: missing closing ) on main()