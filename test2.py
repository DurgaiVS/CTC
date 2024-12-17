# import numpy
import sys
from time import time

import torch
from ctcdecode import CTCBeamDecoder
from tqdm import tqdm
from zspeech.asr.tokenizer import WordPieceTokenizer

from zctc import CTCDecoder


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


if __name__ == "__main__":

    batch_size = 4
    seq_len = 1000
    seq_lens = torch.empty((batch_size), dtype=torch.int32).fill_(seq_len)
    cutoff_top_n = 40
    cutoff_prob = 1.0
    blank_id = 0
    alpha = 0.17
    beta = 0.24
    beam_width = 300
    is_bpe_based = True
    unk_score = -5.0
    lm_path = f"/home/durgai/zspeech/inference/clients/python/zspeech/inference/resources/models/lm/kenlm/model.bin"
    lexicon_fst_path = f"/home/durgai/zspeech/inference/clients/python/zspeech/inference/resources/models/spell_checker/en/lexicon.fst.opt"
    lexicon_fst_path = (
        f"/home/durgai/workspace/zspeech/datasets/spell_checker/lexicon.fst.opt"
    )
    vocab_path = f"/home/durgai/zspeech/datasets/vocab/asr_training/train_wpe_512.txt"
    vocab, apostrophe_id = read_vocab(vocab_path)
    vocab_size = len(vocab)
    tok_sep = "#"
    tokenizer = WordPieceTokenizer(vocab)
    # lm_path = None
    # lexicon_fst_path = None

    old_decoder = CTCBeamDecoder(
        vocab,
        lm_path,
        alpha,
        0,
        cutoff_top_n,
        cutoff_prob,
        beam_width,
        batch_size,
        blank_id,
        True,
        True,
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

    iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    new_latency, old_latency = 0, 0
    hotwords = ["Durgai"]
    hotwords_ids = [tokenizer._tokenize(word).ids for word in hotwords]
    hotwords_tokens = [tokenizer.tokenize(word) for word in hotwords]
    hotwords_weight = [10.0]

    try:
        for _ in tqdm(
            range(iterations), desc=(sys.argv[2] if len(sys.argv) == 3 else "")
        ):

            logits = torch.randn(
                (batch_size, seq_len, vocab_size), device="cpu", dtype=torch.float32
            ).softmax(dim=2)
            argmax = torch.argmax(logits, 2)

            start = time()
            new_labels, _ = new_decoder.decode(
                logits, seq_lens, hotwords_ids, hotwords_weight
            )
            end = time()
            new_latency += end - start

            start = time()
            old_labels, _, _, _ = old_decoder.decode(
                logits, seq_lens, None, hotwords_tokens, hotwords_weight
            )
            end = time()
            old_latency += end - start

            for old_sample, new_sample, args in zip(old_labels, new_labels, argmax):
                for old, new in zip(old_sample, new_sample):

                    old = old[old != 0]
                    new = new[new != 0]

                    assert old.shape[-1] > (seq_len // 2), "Returned only zeros..."
                    assert new.shape[-1] > (seq_len // 2), "Returned only zeros..."

                    assert new.shape[-1] != args.shape[-1] or torch.all(
                        torch.eq(new, args)
                    ), f"Error raised. Args: {args}, Got: {new}"
                    break

    except KeyboardInterrupt:
        print(f"AVG time for NEW CTC after {_ + 1} runs: ", new_latency / (_ + 1))
        print(f"AVG time for OLD CTC after {_ + 1} runs: ", old_latency / (_ + 1))

    else:
        print(
            f"AVG time for NEW CTC after {iterations} runs: ", new_latency / iterations
        )
        print(
            f"AVG time for OLD CTC after {iterations} runs: ", old_latency / iterations
        )
