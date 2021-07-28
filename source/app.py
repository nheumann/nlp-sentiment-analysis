import pandas as pd
import numpy as np
from typing import Any, Optional, List
from seaborn import palettes
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, file_utils, Pipeline, AutoTokenizer, pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D
import seaborn as sns
import spiderplot as sp
from convokit import Corpus, download
from IPython.display import Markdown, display

from .emotion import positive_emotions, negative_emotions, is_positive_emotion, emotion_to_emoji
from .classification_utils import MultiLabelTextClassification




class EmotionPredictor:
    def __init__(self, dataset: pd.DataFrame, model_path: str, filter_less_than: int=1000):
        self.dataset = dataset
        self.person_label = "person"
        self.text_label = "line"
        self.possible_emotions = ["admiration", "amusement",   "anger",    "annoyance",    "approval",    "caring",    "confusion",    "curiosity",    "desire",    "disappointment",    "disapproval",    "disgust",    "embarrassment",    "excitement",
    "fear",    "gratitude",    "grief",    "joy",    "love",    "nervousness",    "optimism",    "pride",    "realization",    "relief",    "remorse",    "sadness",    "surprise",    "neutral"]
        self._initialize_pipeline(model_path)
        self.total_emotions_per_person = None
       
        person_counts = self.dataset["person"].value_counts()
        self.frequent_characters = person_counts[person_counts > filter_less_than].reset_index()["index"].to_numpy()
    
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

    def analyze_persons(self, relevant_persons: List[str], max_number_of_emotions: int=3) -> pd.DataFrame:
        temp_emotions = list()
        for person, emotions in self.total_emotions_per_person.items():
            if person in self.frequent_characters:
                for emotion, value in emotions.items():
                    temp_emotions.append([person, emotion, value])
        df = pd.DataFrame(temp_emotions, columns=["person", "emotion", "value"])

        
        df['avg_value'] = self.analyze_dataset_average(df)
        person_averaged_df = df[np.isin(df['person'], relevant_persons)]
        self.spider_plot(person_averaged_df, relevant_persons)

        for p in person_averaged_df["person"].unique():
            person_df = person_averaged_df[person_averaged_df["person"] == p]
            top_emotions = person_df.sort_values(by=['avg_value'], ascending = False).head(max_number_of_emotions)
            bottom_emotions = person_df.sort_values(by=['avg_value'], ascending = True).head(max_number_of_emotions)
            #unfrequent_emotion_threshold = 0.01
            #top_frequent_emotions = person_df[person_df['value'] > unfrequent_emotion_threshold].sort_values(by=['avg_value'], ascending = False).head(number_of_emotions)
            #bottom_frequent_emotions = person_df[person_df['value'] > unfrequent_emotion_threshold].sort_values(by=['avg_value'], ascending = True).head(number_of_emotions)

            top_emotions = top_emotions[top_emotions["avg_value"] > 1.05]
            
            top_emotions_md = [f'<span style="color: {"green" if is_positive_emotion(row["emotion"]) else "red"}; font-weight:bold; text-transform: uppercase; font-size: larger">{row["emotion"]} {emotion_to_emoji[row["emotion"]]}</span><span style="font-size: larger">({int((row["avg_value"]-1)*100)}% higher)</span>' for _, row in top_emotions.iterrows()]
            top_emotions_md =", ".join(top_emotions_md)
            
            bottom_emotions = bottom_emotions[bottom_emotions["avg_value"] < 0.95]
            bottom_emotions_md = [f'<span style="color: {"green" if is_positive_emotion(row["emotion"]) else "red"}; font-weight:bold; text-transform: uppercase; font-size: smaller">{row["emotion"]} {emotion_to_emoji[row["emotion"]]}</span><span style="color:grey; font-size: smaller">({int((row["avg_value"]-1)*(-100))}% lower)</span>' for _, row in bottom_emotions.iterrows()]
            bottom_emotions_md =", ".join(bottom_emotions_md)
            
            display(Markdown(f'<h2 style=color: black>{p}</h2>'))
            display(Markdown(f'<span style="font-size: larger">Higher ⬆️than average: </span>' + top_emotions_md))        
            display(Markdown(f'<span style="color:grey; font-size: smaller">Lower ⬇️ than average: </span>' + bottom_emotions_md))        
        

    def plot_all_emotions(self, df) -> None:
        sns.set(rc={'figure.figsize':(30,8)})
        sns.set_theme(style="whitegrid")
        chart = sns.barplot(x="emotion", y="value", hue="person", data=df)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        chart.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.show()


    def spider_plot(self, df, relevant_persons) -> None:
        title_size = 30
        tick_size = 15
        figure_size = (20,10)
        padding=0.1
        num_persons = len(df['person'].unique())

        sns.set_style("whitegrid")
        plt.figure(figsize=figure_size)
        plt.title("Differences to average emotions")
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        plt.title('Negative Emotions', fontsize=title_size)
        plt.xticks(fontsize=tick_size)

        colors = sns.color_palette("viridis", num_persons)

        df_negative_emotions= df[np.isin(df['emotion'], negative_emotions)]
        ax = sp.spiderplot(x="emotion", y="avg_value", hue='person', data=df_negative_emotions, rref=1, ax=ax1, palette=colors)
        ax.get_legend().remove()
        ax.set_rlim([min(0.9, df_negative_emotions['avg_value'].min()-padding), max(1.1, df_negative_emotions['avg_value'].max()+padding)])

        ax2 = plt.subplot(1, 2, 2, projection='polar')
        plt.title('Positive Emotions', fontsize=title_size)
        plt.xticks(fontsize=tick_size)

        df_positive_emotions = df[np.isin(df['emotion'], positive_emotions)]
        ax = sp.spiderplot(x="emotion", y="avg_value", hue='person', data=df_positive_emotions, rref=1, ax=ax2, palette=colors)
        
        ax.set_rlim([min(0.9, df_positive_emotions['avg_value'].min()-padding), max(1.1, df_positive_emotions['avg_value'].max()+padding)])

        def update_prop(handle, orig):
            handle.update_from(orig)
            handle.set_marker("s")
        plt.legend(loc='upper center', bbox_to_anchor=(-0.1, -0.1), markerscale=2 ,fancybox=False, shadow=False, ncol=num_persons, fontsize=tick_size, handler_map={plt.Line2D:HandlerLine2D(update_func=update_prop)})

        plt.show()

    def analyze_dataset_average(self, df) -> Any:
        average_emotions = df[['value']].div(df.groupby(by=["emotion"]).transform('mean'))
        average_emotions[average_emotions.isna() | np.isinf(average_emotions)] = 0
        return average_emotions

class EmotionPredictorZeroShot(EmotionPredictor):
    def _initialize_pipeline(self, model_path):
        self.inference_pipeline = pipeline("zero-shot-classification", device=0, model=model_path, tokenizer = model_path)

    def _run_pipeline(self, input_data):
        prediction = self.inference_pipeline(input_data, self.labels, multi_label=True)
        if len(input_data.line) == 1:
            prediction = [prediction]
            prediction = [[{'label' : label, 'score': value} for label, value in zip(sentence['labels'], sentence['scores'])] for sentence in prediction]
        return self._analyze_result(prediction, .8)


