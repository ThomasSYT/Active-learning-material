"""
Class for doing active learning with different sampling strategies.
"""
from scipy import stats
import random

class Active_Learning():
    # Initialize by setting the unlabeled data and the model
    def __init__(self, unlabeled_data, model, randomseed):
        self.unlabeled_data = unlabeled_data
        self.model = model
        self.seed = randomseed
        random.seed(randomseed)

    # Get predictions for the unlabeled data
    def get_predictions(self):
        return self.model.predict_proba(self.unlabeled_data)

    # Uncertainty sampling based on a heuristic: [entroy, confidence, margin]
    def get_most_uncertain(self, heuristic):
        predictions = self.get_predictions()
        most_uncertain = []
        # Return the result with the smallest prediction confidence
        if heuristic == 'confidence':
            certainty = 1.0
            for sample, prediction in zip(self.unlabeled_data, predictions):
                if sorted(prediction)[0] < certainty:
                    most_uncertain = sample
                    certainty = sorted(prediction)[0]
        else:
            most_uncertain = self.get_random()
        return most_uncertain

    def get_random_index(self):
        return random.randint(0, len(self.unlabeled_data))

    def get_random(self):
        return self.unlabeled_data[self.get_random_index()]
