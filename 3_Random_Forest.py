# 3.  Predicting Customer purchase behaviour with Random Forest built from Scratch

import numpy as np

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / self.n_estimators


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        if len(y) <= self.min_samples_split or (self.max_depth is not None and depth == self.max_depth):
            return np.mean(y)

        feature_index, threshold = self.find_best_split(X, y)

        if feature_index is None:
            return np.mean(y)

        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices

        left_subtree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold,
                'left_subtree': left_subtree, 'right_subtree': right_subtree}

    def find_best_split(self, X, y):
        best_feature_index = None
        best_threshold = None
        best_loss = float('inf')

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                if len(y[left_indices]) < self.min_samples_split or len(y[right_indices]) < self.min_samples_split:
                    continue

                loss = self.calculate_loss(y[left_indices], y[right_indices])

                if loss < best_loss:
                    best_loss = loss
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def calculate_loss(self, left_y, right_y):
        left_weight = len(left_y) / (len(left_y) + len(right_y))
        right_weight = len(right_y) / (len(left_y) + len(right_y))

        left_mse = np.mean((left_y - np.mean(left_y))**2)
        right_mse = np.mean((right_y - np.mean(right_y))**2)

        return left_weight * left_mse + right_weight * right_mse

    def predict(self, X):
        return np.array([self.predict_single(x, self.tree) for x in X])

    def predict_single(self, x, tree):
        if isinstance(tree, (int, float)):
            return tree

        if x[tree['feature_index']] < tree['threshold']:
            return self.predict_single(x, tree['left_subtree'])
        else:
            return self.predict_single(x, tree['right_subtree'])
