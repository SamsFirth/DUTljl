from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available
from joyaiSFT.sft.metrics_utils.constants import IGNORE_INDEX
from joyaiSFT.sft.metrics_utils.misc import numpify
from joyaiSFT.sft.metrics_utils.packages import is_rouge_available
if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer
if is_jieba_available():
    import jieba
if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
if is_rouge_available():
    from rouge_chinese import Rouge

def eval_logit_processor(logits: 'torch.Tensor', labels: 'torch.Tensor') -> 'torch.Tensor':
    """Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:
            logits = logits[0]
        else:
            logits = logits[1]
    if logits.dim() != 3:
        raise ValueError('Cannot process the logits.')
    return torch.argmax(logits, dim=-1)

@dataclass
class ComputeSimilarity:
    """Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """
    tokenizer: 'PreTrainedTokenizer'

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, 'score_dict'):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
        self.score_dict = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-1': [], 'bleu-2': [], 'bleu-3': [], 'bleu-4': []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: 'EvalPrediction', compute_result: bool=True) -> Optional[dict[str, float]]:
        preds, labels = (numpify(eval_preds.predictions), numpify(eval_preds.label_ids))
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            if len(' '.join(hypothesis).split()) == 0 or len(' '.join(reference).split()) == 0:
                result = {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
                result = scores[0]
                refs = [reference]
                hyp = hypothesis
                smooth = SmoothingFunction().method3
                bleu1 = sentence_bleu(refs, hyp, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smooth)
                bleu2 = sentence_bleu(refs, hyp, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=smooth)
                bleu3 = sentence_bleu(refs, hyp, weights=(1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=smooth)
                bleu4 = sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            for k, v in result.items():
                self.score_dict[k].append(round(v['f'] * 100, 4))
            self.score_dict['bleu-1'].append(round(bleu1 * 100, 4))
            self.score_dict['bleu-2'].append(round(bleu2 * 100, 4))
            self.score_dict['bleu-3'].append(round(bleu3 * 100, 4))
            self.score_dict['bleu-4'].append(round(bleu4 * 100, 4))
        if compute_result:
            return self._dump()