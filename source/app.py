import pandas as pd
import numpy as np
from typing import Any, Optional
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, file_utils, Pipeline, AutoTokenizer, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from convokit import Corpus, download

from .classification_utils import MultiLabelTextClassification

class EmotionPredictor:
    def __init__(self, dataset: pd.DataFrame, model_path: str):
        self.dataset = dataset
        self.person_label = "person"
        self.text_label = "line"
        self.possible_emotions = ["admiration", "amusement",   "anger",    "annoyance",    "approval",    "caring",    "confusion",    "curiosity",    "desire",    "disappointment",    "disapproval",    "disgust",    "embarrassment",    "excitement",
    "fear",    "gratitude",    "grief",    "joy",    "love",    "nervousness",    "optimism",    "pride",    "realization",    "relief",    "remorse",    "sadness",    "surprise",    "neutral"]
        self._initialize_pipeline(model_path)
        self.total_emotions_per_person = None
    
    def _initialize_pipeline(self, model_path):
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.inference_pipeline = MultiLabelTextClassification(model=model, tokenizer=tokenizer,  return_all_scores=True, device=0)

    @staticmethod
    def _analyze_result(result, threshold = 0.5):
        """Sort the results and throw away all labels with prediction under threshold"""
        output = []
        for sample in result:
            sample = np.array(sample)
            scores = np.array([label['score'] for label in sample])
            predicted_samples = np.argwhere(scores > threshold).reshape(-1)
            output.append(sorted(sample[predicted_samples], key = lambda item: item['score'], reverse=True))
        return output

    def _run_pipeline(self, input_data):
            prediction = self.inference_pipeline(input_data)
            return self._analyze_result(prediction, .2)

    def predict(self, precalculated_predictions: Optional[str]=None) -> None:
        emotions_per_person = {}
        characters = self.dataset[self.person_label]
        for p in characters:
            emotions_per_person[p] = list()

        if precalculated_predictions is not None:
            import pickle
            emotions_per_person = pickle.load(open(precalculated_predictions, 'rb'))
        else:
        # costs time
            for person, text in tqdm(zip(self.dataset[self.person_label], self.dataset[self.text_label]), total=len(self.dataset)):
                result = self._run_pipeline(text)
                result = [(pred['label'], pred['score']) for pred in result[0]] 
                emotions_per_person[person].append(result)

        self.total_emotions_per_person = {}
        for p in characters:
            self.total_emotions_per_person[p] = {}
            for l in self.possible_emotions:
                self.total_emotions_per_person[p][l] = 0

        for person, sentences in emotions_per_person.items():    
            for s in sentences:
                for e in s:
                    self.total_emotions_per_person[person][e[0]] += (e[1]/ len(sentences))

    def analyze_person(self, person: str) -> Any:
        df = pd.DataFrame(columns=["person", "emotion", "value"])

        for person, emotions in self.total_emotions_per_person.items():
            for emotion, value in emotions.items():
                df.loc[len(df)] = [person,emotion,value]
                
        sns.set(rc={'figure.figsize':(30,8)})
        sns.set_theme(style="whitegrid")
        chart = sns.barplot(x="emotion", y="value", hue="person", data=df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        chart.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)



class EmotionPredictorZeroShot(EmotionPredictor):
    def _initialize_pipeline(self, model_path):
        self.inference_pipeline = pipeline("zero-shot-classification", device=0, model=model_path, tokenizer = model_path)

    def _run_pipeline(self, input_data):
        prediction = self.inference_pipeline(input_data, self.labels, multi_label=True)
        if len(input_data.line) == 1:
            prediction = [prediction]
            prediction = [[{'label' : label, 'score': value} for label, value in zip(sentence['labels'], sentence['scores'])] for sentence in prediction]
        return self._analyze_result(prediction, .8)
