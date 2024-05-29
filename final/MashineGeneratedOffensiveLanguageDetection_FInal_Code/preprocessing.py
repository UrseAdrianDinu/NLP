import re
import string

import nltk
from nltk import TweetTokenizer

nltk.download('punkt')

contractions_and_slang = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    # ... (other contractions and slang)
    "xoxo": "love", "thats": "that is", "homie": "best friend",
    "homies": "best friends"
}

def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if w == temp:
            break
        else:
            temp = w
    return w

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def refer_normalize(tokens):
    words = []
    for idx in range(len(tokens)):
        if idx + 1 != len(tokens) and tokens[idx].startswith("@") and tokens[idx + 1].startswith("@"):
            continue
        else:
            words.append(tokens[idx])
    return words

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b(rt|cc)\b', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    pattern = r'[' + re.escape(string.digits + string.punctuation) + ']+'
    text = re.sub(pattern, ' ', text)
    text = remove_emoji(text)
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [contractions_and_slang.get(token, token) for token in tokens]
    clean_text = ' '.join(tokens).strip()
    return clean_text

def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df
