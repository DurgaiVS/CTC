__all__ = ["_Decoder", "ZBeamDecoder"]

import os
import gzip
import shutil
import torch
import numpy
import math
from omegaconf import DictConfig
from typing import Optional
from tempfile import TemporaryDirectory

from .lib._zctc import _Decoder



def get_apostrophe_id_from_vocab(vocab: list[str]) -> int:
    for i, tok in enumerate(vocab):
        if tok == "'": return i

    return -1


class ZBeamDecoder(_Decoder):
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
        lexicon_fst_path: Optional[str] = None
    ):
        apostrophe_id = get_apostrophe_id_from_vocab(vocab)
        lm_alpha = math.log(lm_alpha)

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
            lexicon_fst_path
        )


    def __call__(self, logits: torch.Tensor, seq_lens: torch.Tensor) -> tuple[numpy.ndarray, numpy.ndarray]:

        batch_size = logits.shape[0]
        seq_len = logits.shape[1]

        sorted_indices = torch.argsort(logits, dim=2, descending=True).to(torch.int32)
        
        labels = numpy.zeros((batch_size, self.beam_width, seq_len), dtype=numpy.int32)
        timesteps = numpy.zeros((batch_size, self.beam_width, seq_len), dtype=numpy.int32)

        self.batch_decode(logits.numpy(), sorted_indices.numpy(), labels, timesteps, seq_lens.numpy(), batch_size)

        mask = labels > 0
        return labels[mask], timesteps[mask]


    @classmethod
    def from_cfg(cls, config: DictConfig):

        with TemporaryDirectory() as tmpdir:

            if config.get("lm_path"):
                lm_path: str = config["lm_path"]

                if lm_path[-3:] == ".gz":
                    with gzip.open(lm_path) as f_ip:
                        new_path = os.path.join(tmpdir.name, "kenlm")
                        with open(new_path, "wb") as f_op:
                            shutil.copyfileobj(f_ip, f_op)

                        config["lm_path"] = new_path

            return cls(**config)
