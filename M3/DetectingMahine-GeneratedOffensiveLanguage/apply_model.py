import pandas as pd
from joblib import load

from preprocessing import clean_text

# Load the trained model and vectorizer
best_model = load("trained_model.joblib")
vectorizer = load("vectorizer.joblib")


def predict_human_or_machine(tweet_text):
    """
    Predict whether a tweet was written by a human or a machine.
    """
    # Preprocess the text
    cleaned_text = clean_text(tweet_text)
    # Vectorize the preprocessed text
    tweet_vector = vectorizer.transform([cleaned_text])
    # Predict the label
    prediction = best_model.predict(tweet_vector)[0]
    return prediction


def add_human_machine_column(csv_file_path, output_csv_file_path):
    """
    Add a 'Human/Machine' column to a CSV file based on predictions.
    """
    # Load the CSV file containing the tweets
    tweets_df = pd.read_csv(csv_file_path)

    # Find the column index of 'text'
    text_column_index = None
    for i, col in enumerate(tweets_df.columns):
        if col.lower() == 'text':
            text_column_index = i
            break

    if text_column_index is None:
        raise ValueError("Could not find 'text' column in the CSV file.")

    # Predict 'Human/Machine' label for each tweet
    predictions = []
    for i, row in tweets_df.iterrows():
        tweet_text = row[text_column_index]
        prediction = predict_human_or_machine(tweet_text)
        predictions.append(prediction)

    # Add 'Human/Machine' column to DataFrame
    tweets_df.insert(text_column_index + 1, 'Human/Machine', predictions)

    # Save DataFrame to a new CSV file
    tweets_df.to_csv(output_csv_file_path, index=False)


# Path to the original CSV file
csv_file_path = "new_tweets.csv"

# Path to the output CSV file with predictions
output_csv_file_path = "new_tweets_with_predictions.csv"

try:
    # Add 'Human/Machine' column to the CSV file
    add_human_machine_column(csv_file_path, output_csv_file_path)
    print("Prediction completed. Output saved to:", output_csv_file_path)
except ValueError as e:
    print("Error:", e)
