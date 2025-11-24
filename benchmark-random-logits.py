import gc
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from time import time

import torch
from ctcdecode import CTCBeamDecoder as ParlanceCTCBeamDecoder
from pyctcdecode import BeamSearchDecoderCTC, build_ctcdecoder
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder._ctc_decoder import CTCDecoder as FCTCDecoder
from tqdm import tqdm

from zctc import CTCBeamDecoder

ZCTC_val = 0
PARLANCE_val = 0
KENSHO_val = 0
FL_val = 0
THREAD_COUNT = 0
BATCH_SIZE = 0
BEAM_WIDTH = 0
VOCAB_SIZE = 0
MIN_TOK_PROB = 0
MAX_BEAM_DEVIATION = 0
FORK_POOL = multiprocessing.get_context("fork").Pool


def read_vocab(vocab_path: str):
    vocab = []
    apostrophe_id = -1

    with open(vocab_path) as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            vocab.append(line)

            if line == "'":
                apostrophe_id = i

    return vocab, apostrophe_id


def kensho_ctc(
    decoder: BeamSearchDecoderCTC,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
):

    assert THREAD_COUNT > 0, "THREAD_COUNT should be greater than 0"
    assert BATCH_SIZE > 0, "BATCH_SIZE should be greater than 0"
    assert BEAM_WIDTH > 0, "BEAM_WIDTH should be greater than 0"
    assert MIN_TOK_PROB < 0, "MIN_TOK_PROB should be less than 0"
    assert MAX_BEAM_DEVIATION < 0, "MAX_BEAM_DEVIATION should be less than 0"

    start = time()
    with FORK_POOL(processes=THREAD_COUNT) as pool:
        op = decoder.decode_batch(
            pool,
            [l[:s].cpu().numpy() for l, s in zip(logits, seq_lens)],
            BEAM_WIDTH,
            MAX_BEAM_DEVIATION,
            MIN_TOK_PROB,
        )
    end = time()

    global KENSHO_val
    KENSHO_val += end - start


def zctc_ctc(
    decoder: CTCBeamDecoder,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
):
    logits = logits.exp().to(torch.float64)

    start = time()
    preds, timesteps, seq_pos = decoder.decode(logits, seq_lens)
    end = time()

    global ZCTC_val
    ZCTC_val += end - start


def flashlight_ctc(
    decoder: FCTCDecoder,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
):
    def func(logit, seq_len, vocab_size):
        res = decoder.decoder.decode(logit.data_ptr(), seq_len, vocab_size)
        hyp = decoder._to_hypo(res[: decoder.nbest])
        return hyp

    start = time()
    # res = decoder(logits, seq_lens)

    with ThreadPool(processes=THREAD_COUNT) as pool:
        #     # res = pool.map(decoder, [logits[i].ctypes.data for i in range(BATCH_SIZE)])
        preds = pool.starmap(
            func, [(logits[i], seq_lens[i], VOCAB_SIZE) for i in range(BATCH_SIZE)]
        )

    # with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
    #     executors = [executor.submit(decoder, logits[i].ctypes.data) for i in range(BATCH_SIZE)]
    #     executors = [executor.submit(decoder, logits[i], seq_lens[i]) for i in range(BATCH_SIZE)]
    #     res = [f.result() for f in executors]
    # res = decoder(logits.ctypes.data)
    end = time()

    global FL_val
    FL_val += end - start


def parlance_ctc(
    decoder: CTCBeamDecoder,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
):
    start = time()
    output, scores, timesteps, osl = decoder.decode(logits, seq_lens)
    end = time()

    global PARLANCE_val
    PARLANCE_val += end - start


def infer_decoders(
    parlance_decoder: CTCBeamDecoder,
    zctc_decoder: CTCBeamDecoder,
    kensho_decoder: BeamSearchDecoderCTC,
    flashlight_decoder: FCTCDecoder,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
):
    parlance_ctc(
        parlance_decoder,
        logits,
        seq_lens,
    )
    kensho_ctc(kensho_decoder, logits, seq_lens)
    flashlight_ctc(
        flashlight_decoder,
        logits,
        seq_lens,
    )
    zctc_ctc(zctc_decoder, logits, seq_lens)


if __name__ == "__main__":

    # USAGE: taskset -a -c "last N core ids" python test.py [n_threads] [iterations]
    # NOTE: The reason for using taskset is to restrict the threads to specific cores for
    #       better performance consistency during benchmarking.
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16

    batch_size = n_threads * 4
    seq_len = 3750
    seq_lens = torch.empty((batch_size), dtype=torch.int32).fill_(seq_len)
    cutoff_top_n = 20
    cutoff_prob = 1.0
    blank_id = 0
    alpha = 0
    beta = 0
    beam_width = 25
    is_bpe_based = True
    log_probs_input = True
    unk_score = -5.0
    min_tok_prob = -5.0
    max_beam_deviation = -10.0
    lm_path = None  # arpa or bin file generated from kenlm
    lexicon_fst_path = None  # fst file generated using ZFST
    vocab_path = None  # vocab file
    vocab, apostrophe_id = read_vocab(vocab_path)
    vocab_size = len(vocab)
    tok_sep = "#"
    THREAD_COUNT = n_threads
    BATCH_SIZE = batch_size
    BEAM_WIDTH = beam_width
    VOCAB_SIZE = vocab_size
    MIN_TOK_PROB = min_tok_prob
    MAX_BEAM_DEVIATION = max_beam_deviation
    CURSOR_UP_ONE = "\x1b[1A"
    ERASE_LINE = "\x1b[2K"
    pattern = ((ERASE_LINE + CURSOR_UP_ONE) * 30) + ERASE_LINE
    sub_pattern = ERASE_LINE + CURSOR_UP_ONE

    parlance_decoder = ParlanceCTCBeamDecoder(
        vocab,
        lm_path,
        alpha,
        beta,
        cutoff_top_n,
        cutoff_prob,
        beam_width,
        n_threads,
        blank_id,
        log_probs_input,
        is_bpe_based,
        unk_score,
        "bpe",
        tok_sep,
        lexicon_fst_path,
    )

    zctc_decoder = CTCBeamDecoder(
        n_threads,
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

    kensho_decoder = build_ctcdecoder(
        [""] + vocab[1:], lm_path, alpha=alpha, beta=beta, unk_score_offset=unk_score
    )

    flashlight_decoder = ctc_decoder(
        lexicon=None,
        tokens=vocab + ["|", "<unk>"],
        lm=lm_path,
        nbest=1,
        beam_size=beam_width,
        beam_size_token=cutoff_top_n,
        lm_weight=alpha,
        unk_score=unk_score,
        blank_token="[UNK]",
        word_score=min_tok_prob,
        log_add=True,
    )

    PARLANCE_val = 0
    ZCTC_val = 0
    KENSHO_val = 0
    FL_val = 0

    def printer():
        print(pattern)
        print(
            f"AVG time for PARLANCE CTC after {i+1} runs: {PARLANCE_val / (i + 1):.5f}"
        )
        print(f"AVG time for ZCTC CTC after {i+1} runs: {ZCTC_val / (i + 1):.5f}")
        print(f"AVG time for KENSHO CTC after {i+1} runs: {KENSHO_val / (i + 1):.5f}")
        print(f"AVG time for FLASHLIGHT CTC after {i+1} runs: {FL_val / (i + 1):.5f}")
        print("\n")

        gc.collect()

    for i in tqdm(range(iterations), leave=False):
        logits = torch.randn(
            (batch_size, seq_len, vocab_size), dtype=torch.float32
        ).log_softmax(2)

        infer_decoders(
            parlance_decoder,
            zctc_decoder,
            kensho_decoder,
            flashlight_decoder,
            logits,
            seq_lens,
        )

        printer()
