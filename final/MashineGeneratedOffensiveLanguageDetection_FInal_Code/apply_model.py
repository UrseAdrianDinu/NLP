import pandas as pd
from joblib import load
import xlwings as xw

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
    final_prediction = 1 if prediction == 0 else 0
    return final_prediction


def add_human_machine_column_and_color(csv_file_path, output_excel_file_path):
    """
    Add a 'Human/Machine' column to a CSV file based on predictions,
    and apply colors to Excel cells based on whether the prediction
    matches the actual label.
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

    # Check if there's an 'actual' column
    if 'actual' in tweets_df.columns:
        # Initialize Excel application
        app = xw.App(visible=False)
        workbook = app.books.add()
        worksheet = workbook.sheets[0]

        # Write DataFrame to Excel
        worksheet.range('A1').value = tweets_df

        # Get the used range
        used_range = worksheet.used_range

        # Get the 'Human/Machine' column
        human_machine_column = used_range.columns[-1]

        # Get the 'actual' column
        actual_column = used_range.columns[tweets_df.columns.get_loc('actual')]

        # Compare 'Human/Machine' predictions with 'actual' labels
        for cell in human_machine_column:
            if cell.value == actual_column[cell.row - 1].value:
                # Apply green color if prediction matches actual
                cell.color = (0, 255, 0)  # RGB for green
            else:
                # Apply red color if prediction does not match actual
                cell.color = (255, 0, 0)  # RGB for red

        # Save the Excel file
        workbook.save(output_excel_file_path)
        workbook.close()
        app.quit()
    else:
        # If 'actual' column not found, raise an error
        raise ValueError("Could not find 'actual' column in the CSV file.")


# Path to the original CSV file
csv_file_path = "new_tweets.csv"

# Path to the output Excel file with predictions and colors
output_excel_file_path = "new_tweets_with_predictions.xlsx"

try:
    # Add 'Human/Machine' column to the CSV file and apply colors to Excel cells
    add_human_machine_column_and_color(csv_file_path, output_excel_file_path)
    print("Prediction completed. Output saved to:", output_excel_file_path)
except ValueError as e:
    print("Error:", e)