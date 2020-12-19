from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin


class PseudoLabeler(BaseEstimator, RegressorMixin):

    def __init__(self, model, test, features, target, sample_rate=0.2, seed=42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.test = test
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "test": self.test,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        if self.sample_rate > 0.0:
            augemented_train = self.__create_augmented_train(X, y)
            self.model.fit(
                augemented_train[self.features],
                augemented_train[self.target]
            )
        else:
            self.model.fit(X, y)

        return self

    def __create_augmented_train(self, X, y):
        num_of_samples = int(len(test) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.test[self.features])

        # Add the pseudo-labels to the test set
        augmented_test = test.copy(deep=True)
        augmented_test[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_test = augmented_test.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_test, temp_train])
        return shuffle(augemented_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

target = 'y'
# Preprocess the data
X_train, X_test = train[features], test[features]
y_train = train[target]
# Create the PseudoLabeler with XGBRegressor as the base regressor
model = PseudoLabeler(
    XGBRegressor(nthread=1),
    test,
    features,
    target
)
# Train the model and use it to predict
model.fit(X_train, y_train)
model.predict(X_train)
