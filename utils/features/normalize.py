"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # mean
    features_mean = np.mean(features, 0)

    # sd
    features_deviation = np.std(features, 0)

    # normalization (x-mu)/sd
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # prevent division by 0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
