import numpy as np
from transformers import file_utils, Pipeline, AutoTokenizer, pipeline
# This is pretty much the source code from TextClassificationPipeline, but when i subclassed it
# didnt correctly work(not sure why, didn't try again at the end) so i just copied the whole code.
# We cant use TextClassification itself, since it only outputs the highest label
if file_utils.is_tf_available():
    from transformers.models.auto.modeling_tf_auto import TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

if file_utils.is_torch_available():
    from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

class MultiLabelTextClassification(Pipeline):
    """
    Text classification pipeline using any :obj:`ModelForSequenceClassification`. See the `sequence classification
    examples <../task_summary.html#sequence-classification>`__ for more information.

    This text classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"sentiment-analysis"` (for classifying sequences according to positive or negative
    sentiments).

    If multiple classification labels are available (:obj:`model.config.num_labels >= 2`), the pipeline will run a
    softmax over the results. If there is a single label, the pipeline will run a sigmoid over the result.

    The models that this pipeline can use are models that have been fine-tuned on a sequence classification task. See
    the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=text-classification>`__.
    """

    def __init__(self, return_all_scores: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.check_model_type(
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
            if self.framework == "tf"
            else MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
        )

        self.return_all_scores = return_all_scores


    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) to classify.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **label** (:obj:`str`) -- The label predicted.
            - **score** (:obj:`float`) -- The corresponding probability.

            If ``self.return_all_scores=True``, one such dictionary is returned per label.
        """
        outputs = super().__call__(*args, **kwargs)


        scores = np.exp(outputs) / (1+np.exp(outputs))
        if self.return_all_scores:
            return [
                [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in enumerate(item)]
                for item in scores
            ]
        else:
            return [
                {"label": self.model.config.id2label[item.argmax()], "score": item.max().item()} for item in scores
            ]

def analyze_result(result, threshold = 0.5):
        """Sort the results and throw away all labels with prediction under threshold"""
        output = []
        for sample in result:
            sample = np.array(sample)
            scores = np.array([label['score'] for label in sample])
            predicted_samples = np.argwhere(scores > threshold).reshape(-1)
            output.append(sorted(sample[predicted_samples], key = lambda item: item['score'], reverse=True))
        return output
