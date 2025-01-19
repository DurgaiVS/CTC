import gc
import random
import logging
from pathlib import Path
from time import time
from typing import List

import torch
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm

from zctc import CTCDecoder

NEW_val = 0
OLD_val = 0


def read_vocab(vocab_path: str) -> tuple[list[str], int]:
    vocab = []
    apostrophe_id = -1

    with open(vocab_path) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            vocab.append(line)

            if line == "'":
                apostrophe_id = i

    return vocab, apostrophe_id


def op_graph_path(logits: torch.Tensor) -> List[torch.Tensor]:
    """
    Provided logits should be 3 dimensional, like,
    (1, SeqLen, VocabSize)
    """
    assert logits.dim() == 3, f"Expected Logits of 3 dimension, but got {logits.dim()}"
    op_graph = []

    for sample in logits.argmax(2):
        prev_l = -1
        for i, curr_l in enumerate(sample):
            if prev_l == curr_l:
                sample[i] = 0
            else:
                prev_l = curr_l

        op_graph.append(sample[sample != 0])

    return op_graph


def new_ctc(decoder: CTCDecoder, logits: torch.Tensor, seq_lens: torch.Tensor):

    start = time()
    labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
    end = time()

    global NEW_val
    NEW_val += end - start

    return labels, timesteps


def old_ctc(decoder: CTCBeamDecoder, logits: torch.Tensor, seq_lens: torch.Tensor):

    start = time()
    output, scores, timesteps, out_seq_len = decoder.decode(logits, seq_lens)
    end = time()

    global OLD_val
    OLD_val += end - start

    return output, timesteps, out_seq_len


def infer_both_decoder(old_decoder, new_decoder, logits, seq_lens):
    op_old, ts_old, out_seq_len = old_ctc(old_decoder, logits, seq_lens)
    op_new, ts_new = new_ctc(new_decoder, logits, seq_lens)

    return op_old, ts_old, out_seq_len, op_new, ts_new


if __name__ == "__main__":

    batch_size = 4
    seq_len = 750
    seq_lens = torch.empty((batch_size), dtype=torch.int32).fill_(seq_len)
    cutoff_top_n = 20
    cutoff_prob = 1.0
    blank_id = 0
    alpha = 0.17
    beta = 0.24
    beam_width = 25
    is_bpe_based = True
    log_probs_input = False
    unk_score = -5.0
    lm_path = None # arpa or bin file generated from kenlm
    lexicon_fst_path = None # fst file generated using ZFST
    vocab_path = None # vocab file
    vocab, apostrophe_id = read_vocab(vocab_path)
    vocab_size = len(vocab)
    tok_sep = "#"

    old_decoder = CTCBeamDecoder(
        vocab,
        lm_path,
        alpha,
        beta,
        cutoff_top_n,
        cutoff_prob,
        beam_width,
        batch_size,
        blank_id,
        log_probs_input,
        is_bpe_based,
        unk_score,
        "bpe",
        tok_sep,
        lexicon_fst_path,
    )
    new_decoder = CTCDecoder(
        batch_size,
        blank_id,
        cutoff_top_n,
        cutoff_prob,
        alpha,
        beam_width,
        vocab,
        unk_score,
        tok_sep,
        lm_path,
        lexicon_fst_path,
    )

    # warmups
    for _ in range(1):
        logits = (
            torch.empty((batch_size, seq_len, vocab_size), dtype=torch.float64)
            .normal_(0.7, 3.0)
            .softmax(2)
        )
        op_old, ts_old, out_seq_len, op_new, ts_new = infer_both_decoder(
            old_decoder, new_decoder, logits, seq_lens
        )

    NEW_val, OLD_val = 0, 0

    iterations = 100

    for i in tqdm(range(iterations), leave=False):
        logits = torch.randn((batch_size, seq_len, vocab_size), device="cpu", dtype=torch.float32).softmax(dim=2)
        seq_lens = torch.empty((batch_size), dtype=torch.int32).fill_(logits.shape[-2])

        op_old, ts_old, out_seq_len, op_new, ts_new = infer_both_decoder(
            old_decoder, new_decoder, logits, seq_lens
        )

        for op_s_old, op_s_new, s_l, a_max in zip(
            op_old, op_new, out_seq_len, op_graph_path(logits)
        ):
            s_l = s_l[0]
            op_s_old = op_s_old[0][:s_l]
            op_s_new = op_s_new[0]
            op_s_new = op_s_new[op_s_new != 0]

    print(f"AVG time for OLD CTC after {iterations} runs: ", OLD_val / (iterations))
    print(f"AVG time for NEW CTC after {iterations} runs: ", NEW_val / (iterations))
