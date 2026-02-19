import torch
import pickle
from scipy.optimize import minimize

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, labels):

        def loss(temp):
            scaled = logits / temp
            probs = torch.softmax(torch.tensor(scaled), dim=1)
            nll = -torch.log(probs[range(len(labels)), labels]).mean()
            return nll.item()

        result = minimize(loss, [1.0], bounds=[(0.5, 5)])
        self.temperature = result.x[0]

    def save(self):
        with open("models/temperature.pkl", "wb") as f:
            pickle.dump(self.temperature, f)
