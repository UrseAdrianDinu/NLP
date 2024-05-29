from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TweetClassifier:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):

        X_train_vect = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vect, y_train)
        return self

    def predict(self, X_test):

        X_test_vect = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vect)

    def vectorize_input(self, X):

        return self.vectorizer.transform(X)