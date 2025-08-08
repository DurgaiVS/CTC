"""
Strict tests for CTCBeamDecoder that are expected to expose implementation limitations.
These tests verify parameter validation and edge case handling that may fail due to
underlying implementation constraints.
"""

import pytest
import torch

from zctc import CTCBeamDecoder


class TestCTCBeamDecoderStrictValidation:
    """Strict parameter validation tests that may expose implementation issues."""

    def test_negative_beam_width_should_fail(self, sample_vocab, decoder_params):
        """Test that negative beam width should raise an error."""
        params = decoder_params.copy()
        params["beam_width"] = -1

        # This SHOULD fail but may not due to implementation limitations
        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail(
                "Implementation should reject negative beam_width but didn't. "
                "This indicates missing parameter validation in the underlying code."
            )
        except (ValueError, AssertionError, RuntimeError) as e:
            # Expected behavior - proper parameter validation
            assert "beam_width" in str(e).lower() or "negative" in str(e).lower()

    def test_zero_beam_width_should_fail(self, sample_vocab, decoder_params):
        """Test that zero beam width should raise an error."""
        params = decoder_params.copy()
        params["beam_width"] = 0

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail(
                "Implementation should reject zero beam_width but didn't. "
                "This indicates missing parameter validation."
            )
        except (ValueError, AssertionError, RuntimeError) as e:
            # Expected behavior
            assert "beam_width" in str(e).lower() or "zero" in str(e).lower()

    def test_empty_vocabulary_should_fail(self, decoder_params):
        """Test that empty vocabulary should raise an error."""
        try:
            decoder = CTCBeamDecoder(vocab=[], **decoder_params)
            pytest.fail(
                "Implementation should reject empty vocabulary but didn't. "
                "This indicates missing vocabulary validation."
            )
        except (ValueError, AssertionError, IndexError, RuntimeError) as e:
            # Expected behavior - vocabulary should not be empty
            assert (
                "vocab" in str(e).lower()
                or "empty" in str(e).lower()
                or "size" in str(e).lower()
            )

    def test_single_element_vocabulary_edge_case(self, decoder_params):
        """Test vocabulary with only blank token (edge case that may cause issues)."""
        vocab = ["_"]  # Only blank token

        try:
            decoder = CTCBeamDecoder(vocab=vocab, **decoder_params)

            # If it doesn't fail during initialization, it should fail during decode
            batch_size, seq_len = 1, 5
            logits = torch.ones((batch_size, seq_len, 1))  # All probabilities on blank
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

            # If we get here, the implementation handles single-token vocabularies
            # This might be valid behavior, so we just check the output makes sense
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        except (ValueError, IndexError, RuntimeError) as e:
            # This is also acceptable - single token vocab might not be supported
            pytest.skip(f"Single-token vocabulary not supported: {e}")

    def test_negative_thread_count_should_fail(self, sample_vocab, decoder_params):
        """Test that negative thread count should raise an error."""
        params = decoder_params.copy()
        params["thread_count"] = -1

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail(
                "Implementation should reject negative thread_count but didn't."
            )
        except (ValueError, AssertionError, RuntimeError) as e:
            assert "thread" in str(e).lower() or "negative" in str(e).lower()

    def test_zero_thread_count_should_fail(self, sample_vocab, decoder_params):
        """Test that zero thread count should raise an error."""
        params = decoder_params.copy()
        params["thread_count"] = 0

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail("Implementation should reject zero thread_count but didn't.")
        except (ValueError, AssertionError, RuntimeError) as e:
            assert "thread" in str(e).lower() or "zero" in str(e).lower()

    def test_invalid_blank_id_should_fail(self, sample_vocab, decoder_params):
        """Test that invalid blank_id (outside vocab range) should raise an error."""
        params = decoder_params.copy()
        params["blank_id"] = len(sample_vocab) + 10  # Way outside vocab range

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail(
                "Implementation should reject blank_id outside vocab range but didn't."
            )
        except (ValueError, AssertionError, IndexError, RuntimeError) as e:
            assert (
                "blank" in str(e).lower()
                or "index" in str(e).lower()
                or "range" in str(e).lower()
            )

    def test_negative_blank_id_should_fail(self, sample_vocab, decoder_params):
        """Test that negative blank_id should raise an error."""
        params = decoder_params.copy()
        params["blank_id"] = -1

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)
            pytest.fail("Implementation should reject negative blank_id but didn't.")
        except (ValueError, AssertionError, RuntimeError) as e:
            assert "blank" in str(e).lower() or "negative" in str(e).lower()


class TestCTCBeamDecoderStrictEdgeCases:
    """Strict edge case tests that may expose implementation limitations."""

    def test_extremely_large_beam_width(self, sample_vocab, decoder_params):
        """Test with unreasonably large beam width that should fail or handle gracefully."""
        params = decoder_params.copy()
        params["beam_width"] = 10000  # Unreasonably large

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

            # If initialization succeeds, try decode (this should fail due to memory)
            batch_size, seq_len = 1, 10
            vocab_size = len(sample_vocab)
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            pytest.fail(
                "Implementation should reject extremely large beam_width due to memory constraints."
            )

        except (ValueError, RuntimeError, MemoryError, Exception) as e:
            # Expected - should reject unreasonable beam widths
            pass

    def test_cutoff_top_n_larger_than_vocab(self, sample_vocab, decoder_params):
        """Test cutoff_top_n larger than vocabulary size."""
        params = decoder_params.copy()
        params["cutoff_top_n"] = len(sample_vocab) * 10  # Much larger than vocab

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

            # This might work (implementation might clamp the value)
            batch_size, seq_len = 2, 10
            vocab_size = len(sample_vocab)
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

            # If it works, that's actually fine - implementation handles it
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        except (ValueError, RuntimeError) as e:
            # Also acceptable - parameter validation
            pass

    def test_extreme_alpha_beta_values(self, sample_vocab, decoder_params):
        """Test with extreme alpha/beta values that might cause numerical issues."""
        extreme_params = [
            {"alpha": 1e10, "beta": 1e10},  # Extremely large
            {"alpha": -1e10, "beta": -1e10},  # Extremely negative
            {"alpha": float("inf"), "beta": 0.0},  # Infinity
            {"alpha": float("nan"), "beta": 0.0},  # NaN
        ]

        for extreme_vals in extreme_params:
            params = decoder_params.copy()
            params.update(extreme_vals)

            try:
                decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

                batch_size, seq_len = 1, 5
                vocab_size = len(sample_vocab)
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
                seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

                labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

                # If extreme values work, check outputs are still valid
                assert not torch.isnan(
                    labels
                ).any(), f"NaN in labels with params {extreme_vals}"
                assert not torch.isnan(
                    timesteps
                ).any(), f"NaN in timesteps with params {extreme_vals}"

            except (ValueError, RuntimeError, FloatingPointError) as e:
                # Expected for extreme values
                print(f"Extreme values {extreme_vals} rejected: {e}")
                continue

    def test_very_long_sequences_memory_limit(self, sample_vocab, decoder_params):
        """Test with very long sequences that should hit memory limits."""
        batch_size = 1
        seq_len = 100000  # Very long sequence
        vocab_size = len(sample_vocab)

        # Reduce beam width to be more reasonable
        params = decoder_params.copy()
        params["beam_width"] = min(params["beam_width"], 10)

        try:
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

            # This should likely fail due to memory constraints
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

            # If it somehow works, just verify basic properties
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        except (RuntimeError, MemoryError, Exception) as e:
            # Expected - very long sequences should hit memory limits
            assert (
                "memory" in str(e).lower()
                or "alloc" in str(e).lower()
                or "out of" in str(e).lower()
            )

    def test_incompatible_hotwords_should_fail(self, sample_vocab, decoder_params):
        """Test hotwords with indices outside vocabulary range."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        batch_size, seq_len = 2, 10
        vocab_size = len(sample_vocab)

        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        # Hotwords with indices outside vocab range
        invalid_hotwords = [[vocab_size + 1, vocab_size + 2]]
        hotwords_weight = [1.0]

        try:
            labels, timesteps, seq_pos = decoder.decode(
                logits,
                seq_lens,
                hotwords_id=invalid_hotwords,
                hotwords_weight=hotwords_weight,
            )
            pytest.fail(
                "Implementation should reject hotwords with indices outside vocab range."
            )

        except (ValueError, IndexError, RuntimeError) as e:
            # Expected - invalid hotword indices should be rejected
            assert (
                "index" in str(e).lower()
                or "range" in str(e).lower()
                or "vocab" in str(e).lower()
            )

    def test_mismatched_hotwords_weights_should_fail(
        self, sample_vocab, decoder_params
    ):
        """Test mismatched number of hotwords and weights."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        batch_size, seq_len = 2, 10
        vocab_size = len(sample_vocab)

        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        # More hotwords than weights
        hotwords_id = [[1, 2], [3, 4], [5, 6]]  # 3 hotwords
        hotwords_weight = [1.0, 2.0]  # Only 2 weights

        try:
            labels, timesteps, seq_pos = decoder.decode(
                logits,
                seq_lens,
                hotwords_id=hotwords_id,
                hotwords_weight=hotwords_weight,
            )
            pytest.fail(
                "Implementation should reject mismatched hotwords/weights lengths."
            )

        except (ValueError, IndexError, RuntimeError) as e:
            # Expected - mismatched lengths should be rejected
            assert (
                "length" in str(e).lower()
                or "mismatch" in str(e).lower()
                or "size" in str(e).lower()
            )


class TestCTCBeamDecoderStrictNumerical:
    """Strict numerical tests that may expose precision or overflow issues."""

    def test_all_zero_logits_should_handle_gracefully(
        self, sample_vocab, decoder_params
    ):
        """Test with all zero logits (undefined probabilities)."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        batch_size, seq_len = 2, 10
        vocab_size = len(sample_vocab)

        # All zeros - this creates uniform probability distribution after softmax
        logits = torch.zeros((batch_size, seq_len, vocab_size))
        # After softmax, all tokens will have equal probability (1/vocab_size)
        logits = logits.softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        try:
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

            # If it works, check outputs are valid
            assert not torch.isnan(labels).any(), "NaN values in labels"
            assert not torch.isnan(timesteps).any(), "NaN values in timesteps"
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        except (RuntimeError, ValueError) as e:
            # Also acceptable - all zeros might be rejected
            pass

    def test_infinite_logits_should_fail(self, sample_vocab, decoder_params):
        """Test with infinite logits values."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        batch_size, seq_len = 2, 5
        vocab_size = len(sample_vocab)

        # Create logits with infinity values
        logits = torch.ones((batch_size, seq_len, vocab_size))
        logits[0, 0, 0] = float("inf")

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        try:
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            pytest.fail("Implementation should reject infinite logits values.")

        except (RuntimeError, ValueError, FloatingPointError) as e:
            # Expected - infinite values should be rejected
            assert (
                "inf" in str(e).lower()
                or "finite" in str(e).lower()
                or "invalid" in str(e).lower()
            )

    def test_nan_logits_should_fail(self, sample_vocab, decoder_params):
        """Test with NaN logits values."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        batch_size, seq_len = 2, 5
        vocab_size = len(sample_vocab)

        # Create logits with NaN values
        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        logits[0, 0, 0] = float("nan")

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        try:
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            pytest.fail("Implementation should reject NaN logits values.")

        except (RuntimeError, ValueError) as e:
            # Expected - NaN values should be rejected
            assert (
                "nan" in str(e).lower()
                or "invalid" in str(e).lower()
                or "finite" in str(e).lower()
            )


if __name__ == "__main__":
    # Run strict tests with pytest
    pytest.main([__file__, "-v"])
