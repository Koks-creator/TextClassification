from collections import Counter
from time import perf_counter
import numpy as np
from tensorflow.keras.models import load_model
import re
import string
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle


class Preprocessing:
    def __init__(self):
        self.stop = set(stopwords.words('english'))

    @staticmethod
    def remove_url(text):
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)

    @staticmethod
    def remove_punct(text):
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_stopwords(self, text):
        filtered_words = [word.lower() for word in text.split() if word.lower() not in self.stop]
        return " ".join(filtered_words)

    @staticmethod
    def counter_word(text_col):
        count = Counter()
        for text in text_col:
            for word in text.split():
                count[word] += 1
        return count


classes = ["Neg", "Pos"]
preprocessing_tool = Preprocessing()


def get_prediction(model_path: str, tokinizer_path: str, sents_list: list):
    model = load_model(model_path)
    with open(tokinizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    df = pd.DataFrame({'Text': sents_list})

    df['Text'] = df['Text'].apply(preprocessing_tool.remove_url)
    df['Text'] = df['Text'].apply(preprocessing_tool.remove_punct)
    df['Text'] = df['Text'].apply(preprocessing_tool.remove_stopwords)

    train_sequences = tokenizer.texts_to_sequences(df['Text'])

    train_padded = pad_sequences(train_sequences, maxlen=20, padding="post",
                                 truncating="post")  # padding="post", truncating="post" uzywa zer

    predictions = model.predict(train_padded)
    predictions = [classes[1] if p > 0.5 else classes[0] for p in predictions]
    return predictions


if __name__ == '__main__':
    start = perf_counter()
    # df = pd.read_csv("TestData.csv")

    # sents = df.text.to_list()
    sents = ["I think this movie is well made",
             "Mind blowing tutorial! Look forward for more similar  projects, especially the one you talked about (counter for several exercises at once). Liked + subscribed + bell turned on!",
             "The soundtrack of this movie is terrible",
             "Hi, I really appreciate the effort you put in to break everything down and make it understandable. I want to ask a questions; what does 'image.flags.writeable = False' do exactly? I understand that it improves performance but how does it do this? Many thanks :)",
             "This film was terrible",
             "He is so stupid",
             "Hey Nick. You're my favourite Youtuber of all time. Thank you so much for sharing your knowledge with us. I have learnt so much from you.",
             "I absolutely love your videos my friend. You’ll be absolutely BLOWN away at the implementation this AI can achieve. My page has over 10 animations created with SD, and if you’re not interested, just check out all the other pages that use SD to create animations. What a world we live in. Truly insane!",
             "In my opinion this movie is really bad."]
    print(get_prediction('TextClassification2.h5', 'tokenizer2.pickle', sents))
    print(perf_counter() - start)
    
    #res ['Pos', 'Pos', 'Neg', 'Pos', 'Neg', 'Neg', 'Pos', 'Pos', 'Neg']


