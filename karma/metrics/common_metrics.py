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


@register_metric("exact_match", optional_args=["ignore_case"], default_args={"ignore_case": True})
class ExactMatchMetric(HfMetric):
    def __init__(self, metric_name: str = "exact_match", **kwargs):
        super().__init__(metric_name)
    
    def evaluate(self, predictions, references, **kwargs):
        return self.metric.compute(predictions=predictions, references=references, ignore_case=kwargs.get("ignore_case", True))


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

@register_metric("tokenised_f1")
class TokenisedF1Metric(BaseMetric):
    def __init__(self, metric_name: str = "tokenised_f1", **kwargs):
        super().__init__(metric_name)

    def tokenize(self, text):
        text = text.replace('\n', '').replace('.', '')
        return re.findall(r'\w+', text.lower())
    
    def evaluate(self, predictions, references , **kwargs):
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
        return sum(f1_scores)/len(f1_scores)

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
