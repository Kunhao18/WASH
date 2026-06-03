"""Detector for Waterbag watermarks.

Adapts the KGW z-score detection to Waterbag's multi-key indicator-based
greenlist generation. For each token, iterates over all (key, indicator)
combinations and takes the one that yields the highest green fraction,
since the detector does not know which key/indicator was used at generation.
"""

import torch
import collections
import scipy.stats

from math import sqrt
from nltk.util import ngrams

from ..watermarks.waterbag import WaterbagWatermark


class WaterbagDetector(WaterbagWatermark):
    """Detector for Waterbag watermarks using z-score hypothesis testing."""

    def __init__(
        self,
        *args,
        z_threshold: float = 4.0,
        ignore_repeated_ngrams: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.z_threshold = z_threshold
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        return numer / denom

    def _compute_p_value(self, observed_count, T):
        return scipy.stats.binom.sf(observed_count, T, self.gamma)

    def _get_greenlist_ids_for_detection(self, input_ids, key, indicator):
        """Get greenlist IDs for a single (key, indicator) pair."""
        prf_idx = (self.key_list == key).nonzero(as_tuple=True)[0][0]
        prf = self.prfs[prf_idx].unsqueeze(0)
        key_tensor = key.unsqueeze(0).float()
        indicator_tensor = torch.tensor([indicator], device=self.device)
        greenlist_ids = self._get_greenlist_ids(
            input_ids, self.gamma, prf, self.vocab_size,
            self.prefix_length, key_tensor, indicator_tensor)
        return set(greenlist_ids[0].cpu().tolist())

    def _score_token(self, prefix_ids, target_id):
        """Check if target token is green under any (key, indicator) combination.

        Returns True if the token is in the greenlist for ANY key/indicator pair.
        This is the detection strategy: since we don't know which key/indicator
        was used, we check all and consider the token green if any match.
        """
        input_ids = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        for key in self.key_list:
            for indicator in [0, 1]:
                greenlist = self._get_greenlist_ids_for_detection(input_ids, key, indicator)
                if target_id in greenlist:
                    return True
        return False

    def _score_sequence(self, input_ids: torch.Tensor):
        if len(input_ids) - self.prefix_length < 1:
            raise ValueError(
                f"Must have at least 1 token to score after "
                f"the first {self.prefix_length} prefix tokens."
            )

        token_list = input_ids.cpu().tolist()
        ngram_list = list(ngrams(token_list, self.prefix_length + 1))
        frequencies_table = collections.Counter(ngram_list)

        ngram_to_watermark_lookup = {}
        for ngram_example in frequencies_table.keys():
            prefix = list(ngram_example[:-1])
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self._score_token(prefix, target)

        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            green_token_count = sum(
                freq * outcome
                for freq, outcome in zip(
                    frequencies_table.values(), ngram_to_watermark_lookup.values()
                )
            )

        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        p_value = self._compute_p_value(green_token_count, num_tokens_scored)
        return {
            "num_tokens_scored": num_tokens_scored,
            "num_green_tokens": green_token_count,
            "green_fraction": green_token_count / num_tokens_scored,
            "z_score": z_score,
            "p_value": p_value,
        }

    def detect(self, tokenized_text, z_threshold: float = None) -> dict:
        """Score tokenized text and return detection results."""
        output_dict = self._score_sequence(tokenized_text)
        z_threshold = z_threshold if z_threshold else self.z_threshold
        output_dict["prediction"] = output_dict["z_score"] > z_threshold
        for key, value in output_dict.items():
            if isinstance(value, int):
                output_dict[key] = float(value)
        return output_dict
