dataset = pd.read_csv("data/friends-final-raw.txt", sep="\t")


corpus = Corpus(filename=download("tennis-corpus"))
utterances = list(corpus.iter_utterances())
texts = [t.text for t in utterances]
speakers = [t.get_speaker().id for t in utterances]
dataset = pd.DataFrame(zip(texts, speakers), columns=['line', 'person']) # todo: change name


# split into sentences
dataset["line"] = dataset["line"].str.split(r'[.!?]+\s')
person_counts = dataset["person"].value_counts()
main_characters = person_counts[person_counts > 1000].reset_index()["index"]
print(main_characters)

dataset = dataset[dataset["person"].isin(main_characters)]