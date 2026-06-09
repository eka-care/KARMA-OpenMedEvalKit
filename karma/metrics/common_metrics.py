import evaluate
from sklearn.metrics import f1_score
from collections import Counter
import re
import ast
import pytrec_eval
from typing import Dict, Any
from karma.metrics.base_metric_abs import BaseMetric
from karma.registries.metrics_registry import register_metric


class HfMetric(BaseMetric):
    def __init__(self, metric_name: str, **kwargs):
        super().__init__(metric_name)
        self.metric = evaluate.load(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references)


@register_metric(
    name="bleu",
    optional_args=["max_order", "smooth"],
    default_args={"max_order": 4, "smooth": True},
)
class BleuMetric(HfMetric):
    def __init__(self, metric_name: str = "bleu", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        smooth = kwargs.get("smooth", True)
        references = [[ref] for ref in references]
        return self.metric.compute(
            predictions=predictions, references=references, smooth=smooth
        )


@register_metric(
    "exact_match", optional_args=["ignore_case"], default_args={"ignore_case": True}
)
class ExactMatchMetric(BaseMetric):
    """Exact match metric - computes percentage of predictions that exactly match references."""

    def __init__(self, metric_name: str = "exact_match", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        ignore_case = kwargs.get("ignore_case", True)

        # Ensure all values are strings (handle None, tuples, and other types)
        clean_predictions = []
        for p in predictions:
            if p is None:
                clean_predictions.append("")
            elif isinstance(p, tuple):
                clean_predictions.append(str(p[0]) if p[0] is not None else "")
            else:
                clean_predictions.append(str(p))

        clean_references = []
        for r in references:
            if r is None:
                clean_references.append("")
            elif isinstance(r, tuple):
                clean_references.append(str(r[0]) if r[0] is not None else "")
            else:
                clean_references.append(str(r))

        # Compute exact match
        matches = 0
        for pred, ref in zip(clean_predictions, clean_references):
            if ignore_case:
                pred = pred.lower()
                ref = ref.lower()
            if pred == ref:
                matches += 1

        accuracy = matches / len(clean_predictions) if clean_predictions else 0.0
        return {"exact_match": accuracy}


@register_metric("accuracy")
class AccuracyMetric(BaseMetric):
    """Fallback accuracy metric that compares predictions and references safely."""

    def __init__(self, metric_name: str = "accuracy", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        matches = 0
        total = len(predictions)
        for pred, ref in zip(predictions, references):
            if str(pred).strip() == str(ref).strip():
                matches += 1
        return {"accuracy": matches / total if total > 0 else 0.0}


@register_metric("f1")
class F1Metric(HfMetric):
    def __init__(self, metric_name: str = "f1", **kwargs):
        super().__init__(metric_name)


@register_metric("wer")
class WERMetric(HfMetric):
    def __init__(self, metric_name: str = "wer", **kwargs):
        super().__init__(metric_name)


@register_metric("cer")
class CERMetric(HfMetric):
    def __init__(self, metric_name: str = "cer", **kwargs):
        super().__init__(metric_name)


@register_metric(
    name="rouge",
    optional_args=["rouge_types", "use_stemmer"],
    default_args={"rouge_types": ["rouge1", "rouge2", "rougeL"], "use_stemmer": True},
)
class RougeMetric(HfMetric):
    """
    ROUGE metric for evaluating text generation and summarization.

    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores by default.
    Useful for open-ended QA and text generation tasks.
    """

    def __init__(self, metric_name: str = "rouge", **kwargs):
        super().__init__(metric_name)

    def evaluate(self, predictions, references, **kwargs):
        rouge_types = kwargs.get("rouge_types", ["rouge1", "rouge2", "rougeL"])
        use_stemmer = kwargs.get("use_stemmer", True)
        return self.metric.compute(
            predictions=predictions,
            references=references,
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
        )


@register_metric("tokenised_f1")
class TokenisedF1Metric(BaseMetric):
    def __init__(self, metric_name: str = "tokenised_f1", **kwargs):
        super().__init__(metric_name)

    def tokenize(self, text):
        text = text.replace("\n", "").replace(".", "")
        return re.findall(r"\w+", text.lower())

    def evaluate(self, predictions, references, **kwargs):
        f1_scores = []
        for prediction, reference in zip(predictions, references):
            pred_tokens = self.tokenize(prediction)
            gold_tokens = self.tokenize(reference)

            pred_counts = Counter(pred_tokens)
            gold_counts = Counter(gold_tokens)

            # Compute overlap
            common = pred_counts & gold_counts
            num_same = sum(common.values())

            if num_same == 0:
                f1_scores.append(0.0)
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return sum(f1_scores) / len(f1_scores)


@register_metric("ndcg@k")
class nDCG_at_k(BaseMetric):
    def __init__(self, metric_name: str = "ndcg@k", k_values=[1, 3, 5, 10], **kwargs):
        self.k_values = k_values
        super().__init__(metric_name, **kwargs)

    def evaluate(self, references, predictions, **kwargs) -> Dict[str, Any]:
        # convert to required format
        ## {
        #       q_id: {
        #           "corpus_id": score,
        #           "corpus_id": score
        #       }
        #  }
        qrels = {}
        for reference in references:
            query_id, qrel = list(reference.items())[0]
            qrels[query_id] = qrel

        results = {}
        for prediction in predictions:
            prediction = ast.literal_eval(prediction)
            query_id, result = list(prediction.items())[0]
            results[query_id] = result

        ndcg = {}

        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = 0.0

        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.k_values])

        # Run evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in self.k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]

        for k in self.k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)

        return {"ndcg@k": ndcg}


@register_metric("recall@k")
class recall_at_k(BaseMetric):
    def __init__(self, metric_name: str = "recall@k", k_values=[1, 3, 5, 10], **kwargs):
        self.k_values = k_values
        super().__init__(metric_name, **kwargs)

    def evaluate(self, references, predictions, **kwargs) -> Dict[str, Any]:
        qrels = {}
        for reference in references:
            query_id, qrel = list(reference.items())[0]
            qrels[query_id] = qrel

        results = {}
        for prediction in predictions:
            prediction = ast.literal_eval(prediction)
            query_id, result = list(prediction.items())[0]
            results[query_id] = result

        recall = {}

        for k in self.k_values:
            recall[f"Recall@{k}"] = 0.0

        recall_string = "recall." + ",".join([str(k) for k in self.k_values])

        # Run evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {recall_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in self.k_values:
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

        for k in self.k_values:
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)

        return {"recall@k": recall}


@register_metric("map@k")
class map_at_k(BaseMetric):
    def __init__(self, metric_name: str = "map@k", k_values=[1, 3, 5, 10], **kwargs):
        self.k_values = k_values
        super().__init__(metric_name, **kwargs)

    def evaluate(self, references, predictions, **kwargs) -> Dict[str, Any]:
        qrels = {}
        for reference in references:
            query_id, qrel = list(reference.items())[0]
            qrels[query_id] = qrel

        results = {}
        for prediction in predictions:
            prediction = ast.literal_eval(prediction)
            query_id, result = list(prediction.items())[0]
            results[query_id] = result

        _map = {}

        for k in self.k_values:
            _map[f"MAP@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in self.k_values])

        # Run evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in self.k_values:
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]

        for k in self.k_values:
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)

        return {"map@k": _map}


@register_metric("precision@k")
class precision_at_k(BaseMetric):
    def __init__(
        self, metric_name: str = "precision@k", k_values=[1, 3, 5, 10], **kwargs
    ):
        self.k_values = k_values
        super().__init__(metric_name, **kwargs)

    def evaluate(self, references, predictions, **kwargs) -> Dict[str, Any]:
        qrels = {}
        for reference in references:
            query_id, qrel = list(reference.items())[0]
            qrels[query_id] = qrel

        results = {}
        for prediction in predictions:
            prediction = ast.literal_eval(prediction)
            query_id, result = list(prediction.items())[0]
            results[query_id] = result

        precision = {}

        for k in self.k_values:
            precision[f"Precision@{k}"] = 0.0

        precision_string = "P." + ",".join([str(k) for k in self.k_values])

        # Run evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in self.k_values:
                precision[f"Precision@{k}"] += scores[query_id]["P_" + str(k)]

        for k in self.k_values:
            precision[f"Precision@{k}"] = round(
                precision[f"Precision@{k}"] / len(scores), 5
            )

        return {"precision@k": precision}


@register_metric("silence_accuracy")
class SilenceAccuracyMetric(BaseMetric):
    """
    Metric for evaluating ASR on silent/empty audio datasets.
    Measures what percentage of predictions are correctly empty.
    """

    def __init__(self, metric_name: str = "silence_accuracy", **kwargs):
        super().__init__(metric_name, **kwargs)

    # Filler words / non-speech vocalizations to remove
    FILLER_WORDS = {
        'hmm', 'hm', 'hmmmm', 'hmmm',
        'umm', 'um', 'ummm', 'ummmm',
        'uhh', 'uh', 'uhhh', 'uhhhh',
        'ahh', 'ah', 'ahhh', 'ahhhh',
        'haaa', 'haa', 'ha', 'haaaa',
        'err', 'er', 'errr',
        'erm', 'ermm',
        'mhm', 'mhmm', 'mmm', 'mm', 'mmmm',
        'ugh', 'urgh',
        'ooh', 'oh', 'ohh',
        'eeh', 'eh', 'ehh',
        'ok', 'okay', 'okk', 'okkk', 'okayy',
        'yeah', 'yea', 'yep', 'yup',
        'yes', 'no', 'nope',
    }

    def preprocess(self, text: str) -> str:
        """
        Remove special characters, bracketed content, and filler words.
        - Removes content in [], <>, ()
        - Removes punctuation like . , - ! ? ; : " '
        - Removes filler words like hmm, umm, um, haaa, etc.
        - Returns cleaned text stripped of whitespace
        """
        if text is None:
            return ""

        # Remove content in brackets: [anything], <anything>, (anything)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\(.*?\)', '', text)

        # Remove special characters / punctuation
        text = re.sub(r'[.,\-!?;:"\'\\/]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove filler words (case-insensitive)
        words = text.split()
        words = [w for w in words if w.lower() not in self.FILLER_WORDS]
        text = ' '.join(words).strip()

        return text

    def _is_empty(self, text: str) -> bool:
        """Check if preprocessed text is empty."""
        return len(self.preprocess(text)) == 0

    def evaluate(self, predictions, references, **kwargs) -> float:
        if not predictions:
            return 1.0

        correct = sum(1 for hyp in predictions if self._is_empty(hyp))

        return correct / len(predictions)
