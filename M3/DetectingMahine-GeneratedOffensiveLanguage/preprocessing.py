import re
import string

import nltk
from nltk import TweetTokenizer

nltk.download('punkt')

contractions_and_slang = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
                          "can't've": "cannot have", "'cause": "because", "could've": "could have",
                          "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
                          "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                          "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                          "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                          "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
                          "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                          "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                          "I'll've": "I will have", "I'm": "I am", "I've": "I have",
                          "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                          "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                          "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                          "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                          "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                          "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
                          "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                          "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                          "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                          "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                          "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                          "she's": "she is", "should've": "should have", "shouldn't": "should not",
                          "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
                          "this's": "this is",
                          "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                          "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                          "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                          "they'll've": "they will have", "they're": "they are", "they've": "they have",
                          "to've": "to have", "wasn't": "was not", "we'd": "we would",
                          "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                          "we're": "we are", "we've": "we have", "weren't": "were not",
                          "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                          "what's": "what is", "what've": "what have", "when's": "when is",
                          "when've": "when have", "where'd": "where did", "where's": "where is",
                          "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                          "who's": "who is", "who've": "who have", "why's": "why is",
                          "why've": "why have", "will've": "will have", "won't": "will not",
                          "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                          "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                          "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                          "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                          "you'll've": "you will have", "you're": "you are", "you've": "you have",
                          "u.s.": "united states",
                          "#lol": "laughing out loud", "#lamo": "laughing my ass off",
                          "#rof": "rolling on the floor laughing",
                          "#covfefe": "ironic", "wtf": "what the fuck", "#wtf": "what the fuck",
                          "tbh": "to be honest",
                          "4ward": "forward",
                          "brb": "be right back",
                          "b4": "before",
                          "bfn": "bye for now",
                          "bgd": "background",
                          "btw": "by the way",
                          "br": "best regards",
                          "clk": "click",
                          "da": "the",
                          "deet": "detail",
                          "deets": "details",
                          "dm": "direct message",
                          "f2f": "face to face",
                          "ftl": " for the loss",
                          "ftw": "for the win",
                          "f**k": "fuck",
                          "f**ked": "fucked",
                          "b***ch": "bitch",
                          "kk": "cool cool",
                          "kewl": "cool",
                          "smh": "so much hate",
                          "yaass": "yes",
                          "a$$": "ass",
                          "bby": "baby",
                          "bc": "because",
                          "coz": "because",
                          "cuz": "because",
                          "cause": "because",
                          "cmon": "come on",
                          "cmonn": "come on",
                          "dafuq": "what the fuck",
                          "dafuk": "what the fuck",
                          "dis": "this",
                          "diss": "this",
                          "ma": "my",
                          "dono": "do not know",
                          "donno": "do not know",
                          "dunno": "do not know",
                          "fb": "facebook",
                          "couldnt": "could not",
                          "n": "and",
                          "gtg": "got to go",
                          "yep": "yes",
                          "yw": "you are welcome",
                          "im": "i am",
                          "youre": "you are",
                          "hes": "he is",
                          "shes": "she is",
                          "theyre": "they are",
                          "af": "as fuck",
                          "fam": "family",
                          "fwd": "forward",
                          "ffs": "for fuck sake",
                          "fml": "fuck my life",
                          "lol": "laugh out loud",
                          "lel": "laugh out loud",
                          "lool": "laugh out loud",
                          "lmao": "laugh my ass off",
                          "lmaoo": "laugh my ass off",
                          "omg": "oh my god",
                          "oomg": "oh my god",
                          "omgg": "oh my god",
                          "omfg": "oh my fucking god",
                          "stfu": "shut the fuck up",
                          "awsome": "awesome",
                          "imo": "in my opinion",
                          "imho": "in my humble opinion",
                          "ily": "i love you",
                          "ilyy": "i love you",
                          "ikr": "i know right",
                          "ikrr": "i know right",
                          "idk": "i do not know",
                          "jk": "joking",
                          "lmk": "let me know",
                          "nsfw": "not safe for work",
                          "hehe": "haha",
                          "tmrw": "tomorrow",
                          "yt": "youtube",
                          "hahaha": "haha",
                          "hihi": "haha",
                          "pls": "please",
                          "ppl": "people",
                          "wtf": "what the fuck",
                          "wth": "what teh hell",
                          "obv": "obviously",
                          "nomore": "no more",
                          "u": "you",
                          "ur": "your",
                          "wanna": "want to",
                          "luv": "love",
                          "imma": "i am",
                          "&": "and",
                          "thanx": "thanks",
                          "til": "until",
                          "till": "until",
                          "thx": "thanks",
                          "pic": "picture",
                          "pics": "pictures",
                          "gp": "doctor",
                          "xmas": "christmas",
                          "rlly": "really",
                          "boi": "boy",
                          "boii": "boy",
                          "rly": "really",
                          "whch": "which",
                          "awee": "awsome",
                          "sux": "sucks",
                          "nd": "and",
                          "fav": "favourite",
                          "frnds": "friends",
                          "info": "information",
                          "loml": "love of my life",
                          "bffl": "best friend for life",
                          "gg": "goog game",
                          "xx": "love",
                          "xoxo": "love",
                          "thats": "that is",
                          "homie": "best friend",
                          "homies": "best friends"
                          }


def normalize_word(word):
    temp = word
    while True:
        w = re.sub(r"([a-zA-Z])\1\1", r"\1\1", temp)
        if (w == temp):
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
    """
    Function to preprocess tweet text by removing URLs, emojis, special characters, etc.,
    and expanding contractions using a predefined dictionary.
    """
    text = text.lower()  # Normalize text to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\b(rt|cc)\b', '', text)  # Remove RT and CC
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'@\S+', '', text)  # Remove mentions

    # Remove all digits and punctuation using string.punctuation and string.digits
    pattern = r'[' + re.escape(string.digits + string.punctuation) + ']+'
    text = re.sub(pattern, ' ', text)

    text = re.sub("["
                  u"\U0001F600-\U0001F64F"  # emoticons
                  u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                  u"\U0001F680-\U0001F6FF"  # transport & map symbols
                  u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                  u"\U00002702-\U000027B0"
                  u"\U000024C2-\U0001F251"
                  "]+", '', text, flags=re.UNICODE)  # Remove emojis

    # Tokenization
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [contractions_and_slang.get(token, token) for token in tokens]
    clean_text = ' '.join(tokens).strip()

    return clean_text


def preprocess_data(df):
    """
    Apply text cleaning to a DataFrame column named 'text'.
    """
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

