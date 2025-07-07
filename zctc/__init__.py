__all__ = ["CTCBeamDecoder", "ZFST"]

import logging
from typing import Optional, Tuple, Union

import torch
from _zctc import _ZFST, _Decoder, _Fst


def _get_apostrophe_id_from_vocab(vocab: list[str]) -> int:
    """
    Find the apostrophe token id from the vocabulary.
    This is used in decoding to handle word boundaries correctly.

    Parameters
    ----------
    vocab: list[str]
        Vocabulary of the model.

    Returns
    -------
    apostrophe_id: int
        The index of the apostrophe token in the vocabulary.
        Returns -1 if the apostrophe token is not found.
    """
    for i, tok in enumerate(vocab):
        if tok == "'":
            return i

    return -1


class CTCBeamDecoder(_Decoder):
    """
    A fast and efficient CTC beam decoder with C++ backend.
    This decoder supports hotwords, lexicon FST, and language model scoring,
    where,
        - hotwords are tokens that can be inserted into the decoded sequence
        with a specified weight.
        - lexicon FST is a finite state transducer build from ZFST.
        - language model scoring is done using KenLM build language model.


    Parameters
    ----------
    thread_count: int
        Number of threads to use for decoding.
    blank_id: int
        The blank token id.
    cutoff_top_n: int
        Number of candidate tokens to parse per timeframe (sorted descendingly).
    cutoff_prob: float
        Candidate tokens to consider whose cumulative probability threshold [0, 1]
        reaches this value (sorted descendingly).
    alpha: float
        Language model weight.
    beta: float
        Word insertion weight.
    beam_width: int
        Beam width to use for decoding.
    vocab: list[str]
        Vocabulary of the model.
    unk_lexicon_penalty: float = -5.0
        Penalty to apply for unknown tokens in the lexicon.
    min_tok_prob: float = -5.0
        Minimum probability for a token to consider in a timeframe.
    max_beam_deviation: float = -10.0
        Maximum beam deviation value from the top most beam.
        EG:
            If the top most beam has a score of 10.0, and the
            `max_beam_deviation` is set to -5.0, then the beams
            with scores less than 5.0 will be pruned.
    tok_sep: str = "#"
        Token separator used in vocabulary tokens.
        Eg: "#" in "##a" for BPE tokens.
    lm_path: Optional[str] = None
        Path to KenLM build language model file (either `bin` or `arpa`).
    lexicon_fst_path: Optional[str] = None
        Path to ZFST build lexicon file (either `fst` or `fst.opt`).
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
        hotwords_id: list[list[int]] = [],
        hotwords_weight: Union[float, list[float]] = [],
        hotwords_fst: _Fst = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expecting the logits to be softmaxed and not in log scale.

        Parameters
        ----------
        logits: torch.Tensor
            Input logits from model (batch_size, seq_len, vocab_size),
            can be of type `torch.float32` or `torch.float64`.
        seq_lens: torch.Tensor
            Length of each unpadded sequence in the batch.
        hotwords_id: list[list[int]]
            List of hotword tokens, where each inner list contains the token ids
            of the hotword. If no hotwords are provided, this can be an empty list
            or a list of empty lists.
        hotwords_weight: Union[float, list[float]]
            List of weights for each hotword token or a single weight for all hotword tokens.
            If a single float is provided, it will be used as the weight for all hotwords.
            If a list is provided, it should match the length of `hotwords_id`.
        hotwords_fst: _Fst
            Hotword FST object build using `self.generate_hw_fst` method.

        Returns
        -------
        labels: torch.Tensor
            Decoded labels (batch_size, beam_width, seq_len).
        timesteps: torch.Tensor
            Timesteps of the decoded labels (batch_size, beam_width, seq_len).
        seq_pos: torch.Tensor
            Start index of both labels and timesteps (batch_size, beam_width).

        Raises
        ------
            ValueError: If the shape of `logits` is not (batch_size, seq_len, vocab_size)
                        or if the shape of `seq_lens` is not (batch_size).
            AssertionError: If the vocab size of `logits` does not match the decoder's vocab size.
        """
        if logits.ndim != 3:
            raise ValueError(
                f"Invalid logits shape {logits.shape}, expecting (batch_size, seq_len, vocab_size)"
            )
        if seq_lens.ndim != 1:
            raise ValueError(
                f"Invalid seq_lens shape {seq_lens.shape}, expecting (batch_size)"
            )

        if logits.device.type != "cpu":
            logits = logits.detach().cpu()
        else:
            logits = logits.detach()

        if seq_lens.device.type != "cpu":
            seq_lens = seq_lens.detach().to("cpu", torch.int32)
        else:
            seq_lens = seq_lens.detach().to("cpu", torch.int32)

        batch_size, seq_len, vocab_size = logits.shape
        assert (
            vocab_size == self.vocab_size
        ), f"Vocab size mismatch {vocab_size} != {self.vocab_size}"

        if isinstance(hotwords_weight, float):
            hotwords_weight = [hotwords_weight] * len(hotwords_id)

        # TODO: sort hotwords in descending based on weight, and for 
        #       same weight sort in ascending based on token length.
        sorted_indices = (
            torch.argsort(logits, dim=2, descending=True)
            .to("cpu", torch.int32)
        )
        labels = torch.zeros((batch_size, self.beam_width, seq_len), dtype=torch.int32)
        timesteps = torch.zeros(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        )
        seq_pos = torch.zeros((batch_size, self.beam_width), dtype=torch.int32)

        self.batch_decode(
            logits.data_ptr(),
            logits.element_size(),
            sorted_indices.data_ptr(),
            labels.data_ptr(),
            timesteps.data_ptr(),
            seq_lens.data_ptr(),
            seq_pos.data_ptr(),
            batch_size,
            seq_len,
            hotwords_id,
            hotwords_weight,
            hotwords_fst,
        )

        return labels, timesteps, seq_pos

    def sequential_decode(
        self,
        logits: torch.Tensor,
        seq_lens: torch.Tensor,
        hotwords_id: list[list[int]] = [],
        hotwords_weight: Union[float, list[float]] = [],
        hotwords_fst: _Fst = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expecting the logits to be softmaxed and not in log scale.
        NOTE: This method should only be used with the `DEBUG` mode
              build of the `zctc`.

        Parameters
        ----------
        logits: torch.Tensor
            Input logits from model (batch_size, seq_len, vocab_size),
            can be of type `torch.float32` or `torch.float64`.
        seq_lens: torch.Tensor
            Length of each unpadded sequence in the batch.
        hotwords_id: list[list[int]]
            List of hotword tokens, where each inner list contains the token ids
            of the hotword. If no hotwords are provided, this can be an empty list
            or a list of empty lists.
        hotwords_weight: Union[float, list[float]]
            List of weights for each hotword token or a single weight for all hotword tokens.
            If a single float is provided, it will be used as the weight for all hotwords.
            If a list is provided, it should match the length of `hotwords_id`.
        hotwords_fst: _Fst
            Hotword FST object build using `self.generate_hw_fst` method.

        Returns
        -------
        labels: torch.Tensor
            Decoded labels (batch_size, beam_width, seq_len).
        timesteps: torch.Tensor
            Timesteps of the decoded labels (batch_size, beam_width, seq_len).
        seq_pos: torch.Tensor
            Start index of both labels and timesteps (batch_size, beam_width).

        Raises
        ------
            ValueError: If the shape of `logits` is not (batch_size, seq_len, vocab_size)
                        or if the shape of `seq_lens` is not (batch_size).
            AssertionError: If the vocab size of `logits` does not match the decoder's vocab size.
            AttributeError: If the method is called with the `RELEASE` mode
                            build of the `zctc`.
        """
        if not hasattr(self, "serial_decode"):
            raise AttributeError(
                "This method can only be used with the `DEBUG` mode build of the `zctc`."
            )
        if logits.ndim != 3:
            raise ValueError(
                f"Invalid logits shape {logits.shape}, expecting (batch_size, seq_len, vocab_size)"
            )
        if seq_lens.ndim != 1:
            raise ValueError(
                f"Invalid seq_lens shape {seq_lens.shape}, expecting (batch_size)"
            )

        if logits.device.type != "cpu":
            logits = logits.detach().cpu()
        else:
            logits = logits.detach()

        if seq_lens.device.type != "cpu":
            seq_lens = seq_lens.detach().to("cpu", torch.int32)
        else:
            seq_lens = seq_lens.detach().to("cpu", torch.int32)

        batch_size, seq_len, vocab_size = logits.shape
        assert (
            vocab_size == self.vocab_size
        ), f"Vocab size mismatch {vocab_size} != {self.vocab_size}"

        if isinstance(hotwords_weight, float):
            hotwords_weight = [hotwords_weight] * len(hotwords_id)

        # TODO: sort hotwords in descending based on weight, and for
        #       same weight sort in ascending based on token length.

        sorted_indices = torch.argsort(logits, dim=2, descending=True).to(
            "cpu", torch.int32
        )
        labels = torch.empty((batch_size, self.beam_width, seq_len), dtype=torch.int32)
        timesteps = torch.empty(
            (batch_size, self.beam_width, seq_len), dtype=torch.int32
        )
        seq_pos = torch.empty((batch_size, self.beam_width), dtype=torch.int32)

        self.serial_decode(
            logits.data_ptr(),
            logits.element_size(),
            sorted_indices.data_ptr(),
            labels.data_ptr(),
            timesteps.data_ptr(),
            seq_lens.data_ptr(),
            seq_pos.data_ptr(),
            batch_size,
            seq_len,
            hotwords_id,
            hotwords_weight,
            hotwords_fst,
        )

        return labels, timesteps, seq_pos

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "Override `__call__` method when inheriting from this class"
        )


class ZFST(_ZFST):
    """
    Lexicon FST builder for CTC decoder.

    Parameters
    ----------
    vocab_path: str
        Path to the vocabulary file.
    fst_path: Optional[str] = None
        Path to the output existing build FST file. If not provided,
        the FST will be initialized clean and empty.
    """

    def __init__(self, vocab_path: str, fst_path: Optional[str] = None):
        super().__init__(vocab_path, fst_path)
