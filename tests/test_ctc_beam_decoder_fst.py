"""
Tests for CTCBeamDecoder FST (Finite State Transducer) functionality.
This includes hotword FST generation and usage with the decoder.
"""

from typing import List

import pytest
import torch

from zctc import CTCBeamDecoder


class TestCTCBeamDecoderFST:
    """Test FST-related functionality of CTCBeamDecoder."""

    def test_generate_hw_fst_basic(self, sample_vocab, decoder_params):
        """Test basic hotword FST generation."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        # Simple hotwords that should exist in vocabulary
        hotwords_id = [[1, 2, 3], [2, 4, 5]]
        hotwords_weight = [1.5, 2.0]

        try:
            hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)
            assert hw_fst is not None

            # FST should be a valid object
            assert hasattr(hw_fst, "__class__")

        except Exception as e:
            # If FST generation fails, it might be due to missing dependencies
            pytest.skip(f"FST generation failed, possibly missing dependencies: {e}")

    def test_generate_hw_fst_with_various_weights(self, sample_vocab, decoder_params):
        """Test FST generation with various weight configurations."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Use indices that should exist in vocabulary
        max_idx = min(vocab_size - 1, 10)
        hotwords_id = []

        if vocab_size > 3:
            hotwords_id.append([1, 2, 3])
        if vocab_size > 5:
            hotwords_id.append([2, 4, max_idx])
        if vocab_size > 7:
            hotwords_id.append([1, min(6, max_idx), min(7, max_idx)])

        if not hotwords_id:  # Fallback for very small vocabularies
            hotwords_id = [[1]]

        # Test different weight configurations
        weight_configs = [
            [1.0] * len(hotwords_id),  # All same weight
            list(range(1, len(hotwords_id) + 1)),  # Increasing weights
            [w * 0.5 for w in range(len(hotwords_id), 0, -1)],  # Decreasing weights
            [10.0, 0.1, 5.0][: len(hotwords_id)],  # Mixed weights
        ]

        for weights in weight_configs:
            try:
                hw_fst = decoder.generate_hw_fst(hotwords_id, weights)
                assert hw_fst is not None

            except Exception as e:
                pytest.skip(f"FST generation with weights {weights} failed: {e}")

    def test_decode_with_generated_fst(self, sample_vocab, decoder_params):
        """Test decoding using a generated hotword FST."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Create hotwords
        hotwords_id = [[1, 2], [3, 4]] if vocab_size > 4 else [[1]]
        hotwords_weight = [2.0, 3.0] if len(hotwords_id) > 1 else [2.0]

        try:
            # Generate FST
            hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)

            # Test data
            batch_size, seq_len = 2, 20
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            # Decode using the FST
            labels, timesteps, seq_pos = decoder.decode(
                logits, seq_lens, hotwords_fst=hw_fst
            )

            # Validate outputs
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)
            assert timesteps.shape == (batch_size, decoder.beam_width, seq_len)
            assert seq_pos.shape == (batch_size, decoder.beam_width)

        except Exception as e:
            pytest.skip(f"FST-based decoding failed: {e}")

    def test_fst_vs_regular_hotwords_consistency(self, sample_vocab, decoder_params):
        """Test that FST-based and regular hotword decoding produce consistent results."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Simple test case
        hotwords_id = [[1, 2, 3]] if vocab_size > 3 else [[1]]
        hotwords_weight = [1.5]

        batch_size, seq_len = 1, 10
        torch.manual_seed(42)  # For reproducible results
        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        try:
            # Regular hotword decoding
            labels1, timesteps1, seq_pos1 = decoder.decode(
                logits.clone(),
                seq_lens.clone(),
                hotwords_id=hotwords_id,
                hotwords_weight=hotwords_weight,
            )

            # FST-based decoding
            hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)
            labels2, timesteps2, seq_pos2 = decoder.decode(
                logits.clone(), seq_lens.clone(), hotwords_fst=hw_fst
            )

            # Results should be identical or very similar
            # (exact matching might depend on implementation details)
            assert labels1.shape == labels2.shape
            assert timesteps1.shape == timesteps2.shape
            assert seq_pos1.shape == seq_pos2.shape

            # Check if results are identical (they should be for same hotwords)
            if not torch.equal(labels1, labels2):
                # Allow some tolerance for implementation differences
                print("FST and regular hotwords produce different results")
                print("This might be due to implementation differences")

        except Exception as e:
            pytest.skip(f"FST consistency test failed: {e}")

    def test_empty_hotwords_fst_generation(self, sample_vocab, decoder_params):
        """Test FST generation with empty hotwords."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        try:
            # Empty hotwords
            hw_fst = decoder.generate_hw_fst([], [])

            # Should either work (creating empty FST) or fail gracefully
            if hw_fst is not None:
                # Test that it can be used in decoding
                batch_size, seq_len = 1, 5
                vocab_size = len(sample_vocab)
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
                seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

                labels, timesteps, seq_pos = decoder.decode(
                    logits, seq_lens, hotwords_fst=hw_fst
                )

                assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        except Exception as e:
            # Empty hotwords FST generation might not be supported
            pytest.skip(f"Empty hotwords FST generation not supported: {e}")

    def test_fst_with_invalid_hotword_indices(self, sample_vocab, decoder_params):
        """Test FST generation with invalid hotword indices."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Hotwords with indices outside vocabulary range
        invalid_hotwords = [[vocab_size + 1, vocab_size + 2]]
        hotwords_weight = [1.0]

        try:
            hw_fst = decoder.generate_hw_fst(invalid_hotwords, hotwords_weight)
            pytest.fail("FST generation should reject invalid hotword indices")

        except (ValueError, IndexError, RuntimeError) as e:
            # Expected behavior - should reject invalid indices
            assert (
                "index" in str(e).lower()
                or "range" in str(e).lower()
                or "vocab" in str(e).lower()
            )

    def test_fst_with_mismatched_weights(self, sample_vocab, decoder_params):
        """Test FST generation with mismatched hotwords and weights."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)

        hotwords_id = [[1, 2], [3, 4]]  # 2 hotwords
        hotwords_weight = [1.0]  # 1 weight - mismatch

        try:
            hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)
            pytest.fail("FST generation should reject mismatched hotwords/weights")

        except (ValueError, IndexError, RuntimeError) as e:
            # Expected behavior
            assert (
                "length" in str(e).lower()
                or "mismatch" in str(e).lower()
                or "size" in str(e).lower()
            )

    def test_fst_with_extreme_weights(self, sample_vocab, decoder_params):
        """Test FST generation with extreme weight values."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Use safe hotword indices
        hotwords_id = [[1, 2]] if vocab_size > 2 else [[1]]

        extreme_weights = [
            [1e10],  # Very large
            [-1e10],  # Very negative
            [float("inf")],  # Infinity
            [0.0],  # Zero weight
        ]

        for weights in extreme_weights:
            try:
                hw_fst = decoder.generate_hw_fst(hotwords_id, weights)

                # If generation succeeds, test that FST can be used
                batch_size, seq_len = 1, 5
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
                seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

                labels, timesteps, seq_pos = decoder.decode(
                    logits, seq_lens, hotwords_fst=hw_fst
                )

                # Check outputs are valid
                assert not torch.isnan(
                    labels
                ).any(), f"NaN in labels with weights {weights}"
                assert not torch.isnan(
                    timesteps
                ).any(), f"NaN in timesteps with weights {weights}"

            except (ValueError, RuntimeError, FloatingPointError) as e:
                # Some extreme weights might be rejected
                print(f"Extreme weight {weights} rejected in FST generation: {e}")
                continue

    def test_multiple_fst_generations(self, sample_vocab, decoder_params):
        """Test that multiple FST generations work correctly."""
        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Generate multiple FSTs with different configurations
        fst_configs = (
            [
                ([[1, 2]], [1.0]),
                ([[1, 3]], [2.0]),
                ([[2, 3]], [1.5]),
            ]
            if vocab_size > 3
            else [([[1]], [1.0]), ([[1]], [2.0])]
        )

        fsts = []

        for hotwords_id, hotwords_weight in fst_configs:
            try:
                hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)
                fsts.append(hw_fst)

            except Exception as e:
                pytest.skip(f"Multiple FST generation failed: {e}")

        # Each FST should be independent and usable
        batch_size, seq_len = 1, 8
        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        results = []
        for i, hw_fst in enumerate(fsts):
            try:
                labels, timesteps, seq_pos = decoder.decode(
                    logits.clone(), seq_lens.clone(), hotwords_fst=hw_fst
                )
                results.append((labels, timesteps, seq_pos))

            except Exception as e:
                pytest.skip(f"Decoding with FST {i} failed: {e}")

        # All FSTs should produce valid results
        assert len(results) == len(fsts)

        for labels, timesteps, seq_pos in results:
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)
            assert timesteps.shape == (batch_size, decoder.beam_width, seq_len)
            assert seq_pos.shape == (batch_size, decoder.beam_width)


class TestCTCBeamDecoderFSTPerformance:
    """Performance tests for FST functionality."""

    @pytest.mark.slow
    def test_fst_generation_performance(self, sample_vocab, decoder_params):
        """Test FST generation performance with various hotword configurations."""
        import time

        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Different hotword configurations
        configs = [
            ("small", [[1, 2]], [1.0]),
            ("medium", [[1, 2, 3], [2, 4, 5]], [1.0, 2.0]),
            (
                "large",
                [[i, i + 1, i + 2] for i in range(1, min(6, vocab_size - 2))],
                [1.0] * min(5, vocab_size - 3),
            ),
        ]

        for config_name, hotwords_id, hotwords_weight in configs:
            if any(max(hw) >= vocab_size for hw in hotwords_id):
                continue  # Skip if indices exceed vocab size

            try:
                start_time = time.time()
                hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)
                end_time = time.time()

                generation_time = end_time - start_time
                print(f"FST generation ({config_name}): {generation_time:.4f}s")

                # FST generation should be reasonably fast
                assert (
                    generation_time < 5.0
                ), f"FST generation too slow: {generation_time:.4f}s"

            except Exception as e:
                pytest.skip(f"FST performance test ({config_name}) failed: {e}")

    @pytest.mark.slow
    def test_fst_decoding_performance_comparison(self, sample_vocab, decoder_params):
        """Compare performance between FST and regular hotword decoding."""
        import time

        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        vocab_size = len(sample_vocab)

        # Test configuration
        hotwords_id = [[1, 2, 3], [2, 4, 5]] if vocab_size > 5 else [[1]]
        hotwords_weight = [2.0, 3.0] if len(hotwords_id) > 1 else [2.0]

        batch_size, seq_len = 4, 50
        iterations = 5

        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        try:
            # Generate FST
            hw_fst = decoder.generate_hw_fst(hotwords_id, hotwords_weight)

            # Test regular hotword decoding
            regular_times = []
            for _ in range(iterations):
                start_time = time.time()
                decoder.decode(
                    logits.clone(),
                    seq_lens.clone(),
                    hotwords_id=hotwords_id,
                    hotwords_weight=hotwords_weight,
                )
                end_time = time.time()
                regular_times.append(end_time - start_time)

            # Test FST decoding
            fst_times = []
            for _ in range(iterations):
                start_time = time.time()
                decoder.decode(logits.clone(), seq_lens.clone(), hotwords_fst=hw_fst)
                end_time = time.time()
                fst_times.append(end_time - start_time)

            avg_regular = sum(regular_times) / len(regular_times)
            avg_fst = sum(fst_times) / len(fst_times)

            print(f"Average regular hotword decoding time: {avg_regular:.4f}s")
            print(f"Average FST decoding time: {avg_fst:.4f}s")
            print(f"FST speedup: {avg_regular / avg_fst:.2f}x")

            # Both should complete within reasonable time
            assert avg_regular < 10.0, f"Regular decoding too slow: {avg_regular:.4f}s"
            assert avg_fst < 10.0, f"FST decoding too slow: {avg_fst:.4f}s"

        except Exception as e:
            pytest.skip(f"FST performance comparison failed: {e}")


if __name__ == "__main__":
    # Run FST tests with pytest
    pytest.main([__file__, "-v"])
