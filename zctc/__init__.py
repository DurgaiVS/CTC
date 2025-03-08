__all__ = ["CTCDecoder", "ZFST"]

from typing import Optional, Tuple, Union

import torch
from _zctc import _ZFST, _Decoder


def _get_apostrophe_id_from_vocab(vocab: list[str]) -> int:
    for i, tok in enumerate(vocab):
        if tok == "'":
            return i

    return -1


class CTCDecoder(_Decoder):
    def __init__(
        self,
        thread_count: int,
        blank_id: int,
        cutoff_top_n: int,
        cutoff_prob: float,
        alpha: float,
        beta: float,
        beam_width: int,
        vocab: list[str],
        unk_lexicon_penalty: float = -5.0,
        min_tok_prob: float = -5.0,
        max_beam_deviation: float = -10.0,
        tok_sep: str = "#",
        lm_path: Optional[str] = None,
        lexicon_fst_path: Optional[str] = None,
    ):
        apostrophe_id = _get_apostrophe_id_from_vocab(vocab)
        assert apostrophe_id >= 0, "Cannot find apostrophe from the vocab provided"

        super().__init__(
            thread_count,
            blank_id,
            cutoff_top_n,
            apostrophe_id,
            cutoff_prob,
            alpha,
            beta,
            beam_width,
            unk_lexicon_penalty,
            min_tok_prob,
            max_beam_deviation,
            tok_sep,
            vocab,
            lm_path,
            lexicon_fst_path,
        )

    def decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        hotwords: list[list[int]] = [],
        hotwords_weight: Union[float, list[float]] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expecting the logits to be softmaxed and not in log scale.
        """
        batch_size, seq_len, _ = logits.shape
        if isinstance(hotwords_weight, float):
            hotwords_weight = [hotwords_weight] * len(hotwords)

#TODO : sort hotwords in descending based on weight, and for same weight
#sort in ascending based on token length...

        sorted_indices = (
            torch.argsort(logits, dim=2, descending=True)
            .detach()
            .to("cpu", torch.int32)
            .numpy()
        )
        labels = torch.zeros((batch_size, self.beam_width, seq_len), dtype=torch.int32)
        timesteps = torch.zeros(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        )
        seq_pos = torch.zeros((batch_size, self.beam_width), dtype=torch.int32)

        self.batch_decode(
            logits.detach().cpu().numpy(),
            sorted_indices,
            labels.numpy(),
            timesteps.numpy(),
            seq_lens.detach().to("cpu", torch.int32).numpy(),
            seq_pos.numpy(),
            batch_size,
            seq_len,
            hotwords,
            hotwords_weight,
        )

        return labels, timesteps, seq_pos

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Override `__call__` method when inheriting from this class"
        )


class ZFST(_ZFST):
    def __init__(self, vocab_path: str, fst_path: Optional[str] = None):
        super().__init__(vocab_path, fst_path)
