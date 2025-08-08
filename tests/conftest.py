"""
Pytest fixtures for CTCBeamDecoder tests.
"""

from pathlib import Path
from typing import List

import pytest
import torch

from zctc import CTCBeamDecoder


@pytest.fixture
def sample_vocab():
    """Sample vocabulary for testing."""
    return [
        "",  # blank token
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "'", " ", "##"
    ]


@pytest.fixture
def decoder_params():
    """Default parameters for CTCBeamDecoder initialization."""
    return {
        "thread_count": 4,
        "blank_id": 0,
        "cutoff_top_n": 20,
        "cutoff_prob": 1.0,
        "alpha": 0.17,
        "beta": 0.24,
        "beam_width": 25,
        "unk_lexicon_penalty": -5.0,
        "min_tok_prob": -5.0,
        "max_beam_deviation": -10.0,
        "tok_sep": "#",
        "lm_path": None,
        "lexicon_fst_path": None,
    }


@pytest.fixture
def zctc_decoder(sample_vocab, decoder_params):
    """Create a ZCTC CTCBeamDecoder instance for testing."""
    return CTCBeamDecoder(vocab=sample_vocab, **decoder_params)


@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 4


@pytest.fixture
def seq_len():
    """Default sequence length for testing."""
    return 100


@pytest.fixture
def sample_logits(batch_size, seq_len, sample_vocab):
    """Generate sample logits for testing."""
    vocab_size = len(sample_vocab)
    # Generate random logits and apply log_softmax to simulate model output
    logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)
    return logits.softmax(dim=2)  # Convert to probabilities as expected by zctc


@pytest.fixture
def sample_seq_lens(batch_size, seq_len):
    """Generate sample sequence lengths."""
    return torch.empty((batch_size,), dtype=torch.int32).fill_(seq_len)


@pytest.fixture
def variable_seq_lens(batch_size, seq_len):
    """Generate variable sequence lengths for testing."""
    seq_lens = torch.randint(
        seq_len // 2, seq_len + 1, (batch_size,), dtype=torch.int32
    )
    return seq_lens


@pytest.fixture
def hotwords_data():
    """Sample hotwords data for testing."""
    return {
        "hotwords_id": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],  # Sample token sequences
        "hotwords_weight": [2.0, 1.5, 3.0],  # Corresponding weights
    }


@pytest.fixture
def large_batch_params():
    """Parameters for large batch testing."""
    return {
        "batch_size": 16,
        "seq_len": 500,
    }


@pytest.fixture
def stress_test_params():
    """Parameters for stress testing."""
    return {
        "batch_size": 32,
        "seq_len": 1000,
    }


@pytest.fixture
def different_dtypes():
    """Different data types to test."""
    return [torch.float32, torch.float64]


@pytest.fixture
def edge_case_params():
    """Parameters for edge case testing."""
    return {
        "small_batch": {"batch_size": 1, "seq_len": 10},
        "large_vocab": {"vocab_size": 1000},
        "min_beam": {"beam_width": 1},
        "max_beam": {"beam_width": 100},
    }


@pytest.fixture(scope="session")
def temp_vocab_file(tmp_path_factory):
    """Create a temporary vocabulary file for testing."""
    vocab_content = "\n".join([
        "",  # blank
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "'", " "
    ])
    
    temp_dir = tmp_path_factory.mktemp("vocab")
    vocab_file = temp_dir / "vocab.txt"
    vocab_file.write_text(vocab_content)
    return str(vocab_file)


@pytest.fixture
def performance_config():
    """Configuration for performance testing."""
    return {"warmup_iterations": 5, "test_iterations": 50, "timeout_seconds": 30}
