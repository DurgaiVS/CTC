import gc
from pathlib import Path
from time import time
from typing import List
import multiprocessing

import torch
from ctcdecode import CTCBeamDecoder
from pyctcdecode import build_ctcdecoder, BeamSearchDecoderCTC
from tqdm import tqdm

from zctc import CTCDecoder

ZCTC_val = 0
PARLANCE_val = 0
KENSHO_val = 0
BATCH_SIZE = 0
BEAM_WIDTH = 0
MIN_TOK_PROB = 0
MAX_BEAM_DEVIATION = 0
FORK_POOL = multiprocessing.get_context("fork").Pool

def read_vocab(vocab_path: str):
    vocab = []

    with open(vocab_path) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            vocab.append(line)

    return vocab

def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    col_batch = []
    max_len = max([logits.shape[-2] for logits in batch])
    for sample in batch:
        pad_len = max_len - sample.shape[-2]
        col_batch.append(torch.nn.functional.pad(sample, (0, 0, 0, pad_len)))

    return torch.stack(col_batch)

def yield_batch(filepath: List[Path], batch_size: int):
    for i in range(0, len(filepath), batch_size):
        batch_logits = []
        batch_seqlen = []
        for j in range(i, i + batch_size):
            if j < len(filepath):
                logits, _, _ = torch.load(filepath[j])
                seq_len = torch.empty((1), dtype=torch.int32).fill_(logits.shape[-2])

                batch_logits.append(logits)
                batch_seqlen.append(seq_len)

        yield collate_fn(batch_logits), torch.IntTensor(batch_seqlen)


def zctc_ctc(decoder: CTCDecoder, logits: torch.Tensor, seq_lens: torch.Tensor):
    logits = logits.exp()

    start = time()
    labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
    end = time()

    global ZCTC_val
    ZCTC_val += end - start

    return labels, timesteps

def kensho_ctc(decoder: BeamSearchDecoderCTC, logits: torch.Tensor, seq_lens: torch.Tensor):

    assert BATCH_SIZE > 0, "BATCH_SIZE should be greater than 0"
    assert BEAM_WIDTH > 0, "BEAM_WIDTH should be greater than 0"
    assert MIN_TOK_PROB < 0, "MIN_TOK_PROB should be less than 0"
    assert MAX_BEAM_DEVIATION < 0, "MAX_BEAM_DEVIATION should be less than 0"

    start = time()
    with FORK_POOL(processes=BATCH_SIZE) as pool:
        op = decoder.decode_batch(pool, logits.squeeze().cpu().numpy(), BEAM_WIDTH, MAX_BEAM_DEVIATION, MIN_TOK_PROB)
    end = time()

    global KENSHO_val
    KENSHO_val += end - start

    return op


def parlance_ctc(decoder: CTCBeamDecoder, logits: torch.Tensor, seq_lens: torch.Tensor):

    start = time()
    output, scores, timesteps, out_seq_len = decoder.decode(logits, seq_lens)
    end = time()

    global PARLANCE_val
    PARLANCE_val += end - start

    return output, timesteps, out_seq_len


def infer_decoders(parlance_decoder, zctc_decoder, kensho_decoder, logits, seq_lens):
    op_parlance, ts_parlance, out_seq_len = parlance_ctc(parlance_decoder, logits, seq_lens)
    op_zctc, ts_zctc = zctc_ctc(zctc_decoder, logits, seq_lens)
    op_kensho = kensho_ctc(kensho_decoder, logits, seq_lens)

    return op_parlance, ts_parlance, out_seq_len, op_zctc, ts_zctc, op_kensho



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
    log_probs_input = True
    unk_score = -5.0
    min_tok_prob = -5.0
    max_beam_deviation = -10.0
    lm_path = None # arpa or bin file generated from kenlm
    lexicon_fst_path = None # fst file generated using ZFST
    vocab_path = None # vocab file
    vocab = read_vocab(vocab_path)
    vocab_size = len(vocab)
    tok_sep = "#"
    BATCH_SIZE = batch_size
    BEAM_WIDTH = beam_width
    MIN_TOK_PROB = min_tok_prob
    MAX_BEAM_DEVIATION = max_beam_deviation
    CURSOR_UP_ONE = "\x1b[1A"
    ERASE_LINE = "\x1b[2K"
    pattern = ((ERASE_LINE + CURSOR_UP_ONE) * 10) + ERASE_LINE

    parlance_decoder = CTCBeamDecoder(
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
    zctc_decoder = CTCDecoder(
        batch_size,
        blank_id,
        cutoff_top_n,
        cutoff_prob,
        alpha,
        beta,
        beam_width,
        vocab,
        min_tok_prob,
        max_beam_deviation,
        unk_score,
        tok_sep,
        lm_path,
        lexicon_fst_path,
    )

    kensho_decoder = build_ctcdecoder([""] + vocab[1:], lm_path, alpha=alpha, beta=beta, unk_score_offset=unk_score)


    # warmups
    for _ in range(1):
        logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32).log_softmax(2)

        op_parlance, ts_parlance, out_seq_len, op_zctc, ts_zctc, op = infer_decoders(
            parlance_decoder, zctc_decoder, kensho_decoder, logits, seq_lens
        )

    PARLANCE_err, ZCTC_err = 0, 0
    ZCTC_val, PARLANCE_val = 0, 0
    KENSHO_val, KENSHO_err = 0, 0


    iterations = 100
    interval = 20

    for i in tqdm(range(iterations), leave=False):
        logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32).log_softmax(2)

        op_parlance, ts_parlance, out_seq_len, op_zctc, ts_zctc, op = infer_decoders(
            parlance_decoder, zctc_decoder, kensho_decoder, logits, seq_lens
        )

        if (i + 1) % iterations == 0:
            print(pattern)
            print(f"AVG time for PARLANCE CTC after {i+1} runs: {PARLANCE_val / (i + 1):.5f}")
            print(f"AVG time for ZCTC CTC after {i+1} runs: {ZCTC_val / (i + 1):.5f}")
            print(f"AVG time for KENSHO CTC after {i+1} runs: {KENSHO_val / (i + 1):.5f}")

            gc.collect()
