import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score

from model import TweetClassifier
from preprocessing import preprocess_data
from joblib import dump, load
from sklearn.neural_network import MLPClassifier


def vectorize_text(data, vectorizer=None):
    """
    Converts text data into TF-IDF vectors. This function will be used for detailed printouts.
    If vectorizer is provided, it will be used for transformation.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=1000)
        vectors = vectorizer.fit_transform(data)
    else:
        vectors = vectorizer.transform(data)
    return vectors, vectorizer


if __name__ == '__main__':
    print("Analysis of the data:")
    # Load and preprocess datasets
    datasets = {
        'test': 'test.csv',
        'train': 'train.csv',
        'validation': 'validation.csv'
    }

    classifier = TweetClassifier(max_features=1000)
    test_vectors = None
    test_labels = None
    validation_vectors = None
    validation_labels = None
    df = None
    test_df = None
    best_model = None
    vectorizer = None

    for name, file_path in datasets.items():
        df = pd.read_csv(file_path)
        df = preprocess_data(df)
        print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in the {name} dataset.")

        # Vectorizing the cleaned text for information
        text_vectors, vectorizer = vectorize_text(df['cleaned_text'], vectorizer)
        print(f"Processing the data: Vectorized text shape for {name} dataset: {text_vectors.shape}")
        print(df.head(5))

        # Using the classifier for the actual model operations
        if name == 'train':
            print("Training of the model:")
            train_vectors = text_vectors
            train_labels = df['account.type'].apply(lambda x: 1 if x == 'human' else 0)
            # Define parameter grid for grid search
            param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (50, 30, 10)], 'max_iter': [300, 400, 500]}
            # Initialize GridSearchCV with MLPClassifier and parameter grid
            grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='accuracy')
            # Perform grid search to find the best parameters
            grid_search.fit(train_vectors, train_labels)
            # Get the best model after grid search
            best_model = grid_search.best_estimator_

            # Specify the number of epochs
            num_epochs = 100

            for epoch in range(1, num_epochs + 1):
                print(f"Epoch {epoch}/{num_epochs}")
                # Train the model on the entire training set for this epoch
                best_model.fit(train_vectors, train_labels)
                print(f"Epoch {epoch}/{num_epochs} - Done")

            # Save the trained model and vectorizer
            dump(best_model, "trained_model.joblib")
            dump(vectorizer, "vectorizer.joblib")

        elif name == 'test':
            test_df = preprocess_data(df)  # Apply preprocessing to test data
            test_vectors, _ = vectorize_text(test_df['cleaned_text'], vectorizer)  # Use the loaded vectorizer
            test_labels = test_df['account.type'].apply(lambda x: 1 if x == 'human' else 0)

        elif name == 'validation':
            validation_vectors, _ = vectorize_text(df['cleaned_text'], vectorizer)  # Use the loaded vectorizer
            validation_labels = df['account.type'].apply(lambda x: 1 if x == 'human' else 0)

    # Ensure we have the test vectors and labels to proceed with prediction and evaluation
    if best_model is not None and test_vectors is not None and test_labels is not None:
        # Predicting and evaluating the test dataset using the classifier
        print("Outputting results for the test dataset:")
        # Load the vectorizer used during training
        vectorizer = load("vectorizer.joblib")
        test_predictions = best_model.predict(test_vectors)
        print("Accuracy on test data:", accuracy_score(test_labels, test_predictions))
        print("Classification Report:\n", classification_report(test_labels, test_predictions))
    else:
        print("Test data not loaded properly.")

    # Model evaluation on validation dataset
    if best_model is not None and validation_vectors is not None and validation_labels is not None:
        print("Evaluating the model on the validation dataset:")
        validation_predictions = best_model.predict(validation_vectors)
        print("Accuracy on validation data:", accuracy_score(validation_labels, validation_predictions))
        print("Classification Report:\n", classification_report(validation_labels, validation_predictions))
    else:
        print("Validation data not loaded properly.")