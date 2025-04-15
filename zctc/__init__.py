__all__ = ["CTCDecoder", "ZFST"]

import logging
from typing import Optional, Tuple, Union

import torch
from _zctc import _ZFST, _Decoder


def _get_apostrophe_id_from_vocab(vocab: list[str]) -> int:
    for i, tok in enumerate(vocab):
        if tok == "'":
            return i

    return -1


class CTCDecoder(_Decoder):
    """
    A fast and efficient CTC beam decoder with C++ backend.

    Args:
        thread_count: Number of threads to use for decoding.
        blank_id: The blank token id.
        cutoff_top_n: Number of candidate tokens to parse per timeframe (sorted descendingly).
        cutoff_prob: Candidate tokens to consider whose cumulative probability threshold [0, 1]
                     reaches this value (sorted descendingly).
        alpha: Language model weight.
        beta: Word insertion weight.
        beam_width: Beam width to use for decoding.
        vocab: Vocabulary of the model.
        unk_lexicon_penalty: Penalty to apply for unknown tokens in the lexicon.
        min_tok_prob: Minimum probability for a token to consider in a timeframe.
        max_beam_deviation: Maximum beam deviation value from the top most beam.
        tok_sep: Token separator used in vocabulary tokens.
                 Eg: "#" in "##a" for BPE tokens.
        lm_path: Path to KenLM build language model file (either `bin` or `arpa`).
        lexicon_fst_path: Path to ZFST build lexicon file (either `fst` or `fst.opt`).
    """

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
        if apostrophe_id < 0:
            logging.warning("Cannot find apostrophe token from the vocab provided")

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

        Args:
            logits: Input logits from model (batch_size, seq_len, vocab_size),
                    can be of type `torch.float32` or `torch.float64`.
            seq_lens: Length of each unpadded sequence in the batch.
            hotwords: List of hotword tokens.
            hotwords_weight: List of weights for each hotword token or a single weight
                             for all hotword tokens.

        Returns:
            A tuple containing the following:
                - labels: Decoded labels (batch_size, beam_width, seq_len).
                - timesteps: Timesteps of the decoded labels (batch_size, beam_width, seq_len).
                - seq_pos: Start index of both labels and timesteps (batch_size, beam_width).
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
            .contiguous()
            .numpy()
        )
        labels = torch.empty((batch_size, self.beam_width, seq_len), dtype=torch.int32)
        timesteps = torch.empty(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        )
        seq_pos = torch.empty((batch_size, self.beam_width), dtype=torch.int32)


        if not logits.is_contiguous():
            logits = logits.contiguous()
        if not seq_lens.is_contiguous():
            seq_lens = seq_lens.contiguous()
        if not labels.is_contiguous():
            labels = labels.contiguous()
        if not timesteps.is_contiguous():
            timesteps = timesteps.contiguous()
        if not seq_pos.is_contiguous():
            seq_pos = seq_pos.contiguous()


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

    def sequential_decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        hotwords: list[list[int]] = [],
        hotwords_weight: Union[float, list[float]] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expecting the logits to be softmaxed and not in log scale.

        NOTE: This method should only be used with the `DEBUG` mode
            build of the `zctc`.

        Args:
            logits: Input logits from model (batch_size, seq_len, vocab_size),
                    can be of type `torch.float32` or `torch.float64`.
            seq_lens: Length of each unpadded sequence in the batch.
            hotwords: List of hotword tokens.
            hotwords_weight: List of weights for each hotword token or a single weight
                             for all hotword tokens.

        Returns:
            A tuple containing the following:
                - labels: Decoded labels (batch_size, beam_width, seq_len).
                - timesteps: Timesteps of the decoded labels (batch_size, beam_width, seq_len).
                - seq_pos: Start index of both labels and timesteps (batch_size, beam_width).

        Raises:
            AttributeError: If the method is called with the `RELEASE` mode
                build of the `zctc`.
        """
        if not hasattr(self, "serial_decode"):
            raise AttributeError(
                "This method can only be used with the `DEBUG` mode build of the `zctc`."
            )

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

        self.serial_decode(
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
    """
    Lexicon FST builder for CTC decoder.

    Args:
        vocab_path: Path to the vocabulary file.
        fst_path: Path to the output FST file.
    """

    def __init__(self, vocab_path: str, fst_path: Optional[str] = None):
        super().__init__(vocab_path, fst_path)
