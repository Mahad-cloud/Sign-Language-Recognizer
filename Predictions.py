import math


class Predictions:

    def __init__(self, feature_first, feature_second, feature_third, feature_fourth, feature_fifth, target,
                 cluster_class):
        self.feature_first = feature_first
        self.feature_second = feature_second
        self.feature_third = feature_third
        self.feature_fourth = feature_fourth
        self.feature_fifth = feature_fifth
        self.target = target
        self.cluster_class = cluster_class

    def findingDistance(train, test):
        distance_1 = (train.feature_first - test.feature_first) ** 2
        distance_2 = (train.feature_second - test.feature_second) ** 2
        distance_3 = (train.feature_third - test.feature_third) ** 2
        distance_4 = (train.feature_fourth - test.feature_fourth) ** 2
        distance_5 = (train.feature_fifth - test.feature_fifth) ** 2
        totalDistance = math.sqrt(distance_1 + distance_2 + distance_3 + distance_4 + distance_5)
        return totalDistance

    def Calculate_Sol(training_outputs, weights):
        sol = 0
        for i in range(0, len(training_outputs)):
            sol = sol + (float(weights[i]) * float(training_outputs[i]))
        return sol

    def Punctuation(string):

        # punctuation marks
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        # traverse the given string and if any punctuation
        # marks occur replace it with null
        for x in string.lower():
            if x in punctuations:
                string = string.replace(x, "")

                # Print string without punctuation
        return string
