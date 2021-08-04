import pandas as pd
from convokit import Corpus, download

def load_friends_dataset(path):
    dataset = pd.read_csv("data/friends-final-raw.txt", sep="\t")
    return dataset

def load_tennis_dataset():
    corpus = Corpus(filename=download("tennis-corpus"))
    utterances = list(corpus.iter_utterances())
    texts = [t.text for t in utterances]
    speakers = [t.get_speaker().id for t in utterances]
    dataset = pd.DataFrame(zip(texts, speakers), columns=['line', 'person'])
    return dataset


def split_multiline_sentence(dataset, line_col:str="line"):
    dataset["line"] = dataset["line"].str.split(r'[.!?]+\s')
    return dataset