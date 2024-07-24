__all__ = ["CTCDecoder", "ZFST"]

from typing import Optional, Union

import numpy
import torch
from _zctc import _ZFST, _Decoder
from omegaconf import DictConfig
from registrable import Registrable


def get_apostrophe_id_from_vocab(vocab: list[str]) -> int:
    for i, tok in enumerate(vocab):
        if tok == "'":
            return i

    return -1


class CTCDecoder(Registrable, _Decoder):
    def __init__(
        self,
        thread_count: int,
        blank_id: int,
        cutoff_top_n: int,
        cutoff_prob: float,
        lm_alpha: float,
        beam_width: int,
        vocab: list[str],
        unk_score: float = -5.0,
        tok_sep: str = "#",
        lm_path: Optional[str] = None,
        lexicon_fst_path: Optional[str] = None,
    ):
        apostrophe_id = get_apostrophe_id_from_vocab(vocab)
        assert apostrophe_id >= 0, "Cannot find apostrophe from the vocab provided"

        super().__init__(
            thread_count,
            blank_id,
            cutoff_top_n,
            apostrophe_id,
            cutoff_prob,
            lm_alpha,
            beam_width,
            unk_score,
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
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
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
            .cpu()
            .numpy()
            .astype(numpy.int32)
        )
        labels = torch.zeros(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        ).numpy()
        timesteps = torch.zeros(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        ).numpy()

        self.batch_decode(
            logits.cpu().numpy(),
            sorted_indices,
            labels,
            timesteps,
            seq_lens.cpu().numpy().astype(numpy.int32),
            batch_size,
            seq_len,
            hotwords,
            hotwords_weight,
        )

        return labels, timesteps

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Override `__call__` method when inheriting from this class"
        )

    @classmethod
    def from_cfg(cls, cfg: DictConfig, *args, **kwargs):
        subcls = cls.by_name(cfg.name)
        return subcls.from_cfg(cfg, *args, **kwargs)


class ZFST(_ZFST):
    def __init__(self, fst_path: Optional[str] = None):
        super().__init__(fst_path)
