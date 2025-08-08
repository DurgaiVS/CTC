"""
Integration tests for CTCBeamDecoder with real-world scenarios.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from zctc import CTCBeamDecoder


class TestCTCBeamDecoderIntegration:
    """Integration tests using real-world scenarios."""

    @pytest.fixture
    def realistic_vocab(self):
        """Create a realistic vocabulary similar to what would be used in practice."""
        # Simulate a vocabulary with common English characters and subwords
        vocab = [
            "",  # blank token at index 0
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "'", " ",  # apostrophe and space
            "##a", "##e", "##i", "##o", "##u",  # common subword tokens
            "##s", "##t", "##n", "##r", "##l",
            "##ing", "##ed", "##er", "##ly", "##ion",  # common suffixes
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one"  # common words
        ]
        return vocab

    @pytest.fixture
    def realistic_decoder_params(self):
        """Realistic decoder parameters based on typical use cases."""
        return {
            "thread_count": 4,
            "blank_id": 0,
            "cutoff_top_n": 40,  # Higher for better quality
            "cutoff_prob": 1.0,
            "alpha": 0.2,  # Language model weight
            "beta": 0.3,  # Word insertion weight
            "beam_width": 100,  # Larger beam for better quality
            "unk_lexicon_penalty": -10.0,
            "min_tok_prob": -8.0,
            "max_beam_deviation": -15.0,
            "tok_sep": "#",
            "lm_path": None,
            "lexicon_fst_path": None,
        }

    @pytest.fixture
    def realistic_decoder(self, realistic_vocab, realistic_decoder_params):
        """Create a realistic decoder instance."""
        return CTCBeamDecoder(vocab=realistic_vocab, **realistic_decoder_params)

    def test_speech_recognition_simulation(self, realistic_decoder):
        """Simulate a typical speech recognition scenario."""
        # Typical speech recognition batch
        batch_size = 8
        seq_len = 500  # ~5 seconds at 100 fps
        vocab_size = realistic_decoder.vocab_size

        # Create more realistic logits (lower entropy, some structure)
        torch.manual_seed(42)  # For reproducible test

        # Generate logits with realistic distribution
        # Blank token should have higher probability at many timesteps
        logits = torch.randn((batch_size, seq_len, vocab_size))

        # Boost blank token probability (common in real CTC outputs)
        logits[:, :, 0] += 1.0  # Boost blank token

        # Add some structure - boost certain tokens at certain times
        for b in range(batch_size):
            for t in range(0, seq_len, 50):  # Every 0.5 seconds
                if t + 10 < seq_len:
                    # Boost some non-blank tokens to simulate speech
                    token_ids = torch.randint(1, min(20, vocab_size), (5,))
                    logits[b, t : t + 10, token_ids] += 0.5

        logits = logits.softmax(dim=2)  # Convert to probabilities

        # Variable sequence lengths
        seq_lens = torch.randint(
            seq_len // 2, seq_len + 1, (batch_size,), dtype=torch.int32
        )

        # Decode
        labels, timesteps, seq_pos = realistic_decoder.decode(logits, seq_lens)

        # Validate outputs
        assert labels.shape == (batch_size, realistic_decoder.beam_width, seq_len)
        assert timesteps.shape == (batch_size, realistic_decoder.beam_width, seq_len)
        assert seq_pos.shape == (batch_size, realistic_decoder.beam_width)

        # Check that we got reasonable outputs
        for b in range(batch_size):
            for beam in range(realistic_decoder.beam_width):
                start_idx = seq_pos[b, beam].item()
                if start_idx > 0:  # Valid beam
                    # Should have some non-blank tokens
                    beam_labels = labels[b, beam, start_idx:]
                    non_blank_count = (beam_labels != 0).sum().item()
                    assert (
                        non_blank_count > 0
                    ), f"Beam {beam} in batch {b} has no non-blank tokens"

    def test_with_hotwords_realistic(self, realistic_decoder, realistic_vocab):
        """Test with realistic hotwords scenario."""
        batch_size = 4
        seq_len = 200
        vocab_size = len(realistic_vocab)

        # Find indices of common words in vocabulary for hotwords
        hotwords_text = ["the", "and", "you"]
        hotwords_id = []

        for word in hotwords_text:
            if word in realistic_vocab:
                word_id = realistic_vocab.index(word)
                hotwords_id.append([word_id])  # Single token hotwords

        if not hotwords_id:
            pytest.skip("No hotwords found in vocabulary")

        hotwords_weight = [3.0, 2.5, 2.0]  # Different weights for different hotwords

        # Generate logits with some bias towards hotword tokens
        torch.manual_seed(123)
        logits = torch.randn((batch_size, seq_len, vocab_size))

        # Slightly boost hotword tokens
        for hotword_list in hotwords_id:
            for token_id in hotword_list:
                logits[:, :, token_id] += 0.3

        logits = logits.softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        # Decode with hotwords
        labels_hw, timesteps_hw, seq_pos_hw = realistic_decoder.decode(
            logits, seq_lens, hotwords_id=hotwords_id, hotwords_weight=hotwords_weight
        )

        # Decode without hotwords for comparison
        labels_no_hw, timesteps_no_hw, seq_pos_no_hw = realistic_decoder.decode(
            logits, seq_lens
        )

        # Both should work and produce valid outputs
        assert labels_hw.shape == labels_no_hw.shape
        assert timesteps_hw.shape == timesteps_no_hw.shape
        assert seq_pos_hw.shape == seq_pos_no_hw.shape

        # Results might be different due to hotwords (but not necessarily)
        print("Hotwords integration test completed successfully")

    def test_batch_processing_workflow(self, realistic_decoder):
        """Test typical batch processing workflow."""
        batches = [
            {"batch_size": 4, "seq_len": 300},
            {"batch_size": 8, "seq_len": 150},
            {"batch_size": 2, "seq_len": 600},
            {"batch_size": 16, "seq_len": 100},
        ]

        vocab_size = realistic_decoder.vocab_size
        all_results = []

        for i, batch_config in enumerate(batches):
            batch_size = batch_config["batch_size"]
            seq_len = batch_config["seq_len"]

            print(f"Processing batch {i+1}: {batch_size} samples, {seq_len} timesteps")

            # Generate batch data
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

            # Decode
            labels, timesteps, seq_pos = realistic_decoder.decode(logits, seq_lens)

            # Store results
            batch_result = {
                "batch_id": i,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "labels": labels,
                "timesteps": timesteps,
                "seq_pos": seq_pos,
            }
            all_results.append(batch_result)

            # Validate this batch
            assert labels.shape == (batch_size, realistic_decoder.beam_width, seq_len)

        # All batches processed successfully
        assert len(all_results) == len(batches)
        print(
            f"Successfully processed {len(batches)} batches with different configurations"
        )

    def test_long_running_session(self, realistic_decoder):
        """Test decoder stability over many decode operations."""
        batch_size = 4
        seq_len = 200
        vocab_size = realistic_decoder.vocab_size

        num_iterations = 50
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        successful_decodes = 0

        for i in range(num_iterations):
            try:
                # Generate new logits for each iteration
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)

                # Decode
                labels, timesteps, seq_pos = realistic_decoder.decode(logits, seq_lens)

                # Basic validation
                assert labels.shape == (
                    batch_size,
                    realistic_decoder.beam_width,
                    seq_len,
                )
                assert timesteps.shape == (
                    batch_size,
                    realistic_decoder.beam_width,
                    seq_len,
                )
                assert seq_pos.shape == (batch_size, realistic_decoder.beam_width)

                successful_decodes += 1

                if (i + 1) % 10 == 0:
                    print(f"Completed {i+1}/{num_iterations} iterations")

            except Exception as e:
                pytest.fail(f"Decode failed at iteration {i+1}: {str(e)}")

        assert successful_decodes == num_iterations
        print(
            f"Successfully completed {successful_decodes} decode operations in long-running session"
        )

    def test_memory_stability_over_time(self, realistic_decoder):
        """Test that memory usage remains stable over many operations."""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

        batch_size = 4
        seq_len = 150
        vocab_size = realistic_decoder.vocab_size
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        # Record initial memory
        initial_memory = process.memory_info().rss

        num_iterations = 100
        memory_samples = []

        for i in range(num_iterations):
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            labels, timesteps, seq_pos = realistic_decoder.decode(logits, seq_lens)

            # Clear references
            del labels, timesteps, seq_pos, logits

            # Sample memory every 10 iterations
            if (i + 1) % 10 == 0:
                current_memory = process.memory_info().rss
                memory_samples.append(current_memory)

                if len(memory_samples) >= 5:
                    # Check that memory isn't consistently growing
                    recent_growth = memory_samples[-1] - memory_samples[-5]
                    growth_mb = recent_growth / (1024 * 1024)

                    # Allow some growth but not excessive
                    assert (
                        growth_mb < 50
                    ), f"Memory grew by {growth_mb:.1f}MB in recent iterations"

        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        total_growth_mb = total_growth / (1024 * 1024)

        print(
            f"Total memory growth over {num_iterations} iterations: {total_growth_mb:.1f}MB"
        )

        # Memory growth should be reasonable
        assert (
            total_growth_mb < 100
        ), f"Total memory growth too high: {total_growth_mb:.1f}MB"

    def test_error_recovery(self, realistic_vocab, realistic_decoder_params):
        """Test that decoder can recover from various error conditions."""
        decoder = CTCBeamDecoder(vocab=realistic_vocab, **realistic_decoder_params)

        batch_size = 4
        seq_len = 100
        vocab_size = len(realistic_vocab)

        # Test 1: Invalid input followed by valid input
        try:
            # Try with wrong vocab size (should fail)
            wrong_logits = torch.randn((batch_size, seq_len, vocab_size + 10)).softmax(
                dim=2
            )
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)
            decoder.decode(wrong_logits, seq_lens)
            pytest.fail("Should have failed with vocab size mismatch")
        except AssertionError:
            pass  # Expected error

        # Now try with correct input (should work)
        correct_logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        labels, timesteps, seq_pos = decoder.decode(correct_logits, seq_lens)

        assert labels.shape == (batch_size, decoder.beam_width, seq_len)
        print("Decoder recovered successfully after error")

        # Test 2: Multiple error conditions
        error_scenarios = [
            # Wrong number of dimensions
            lambda: decoder.decode(torch.randn((batch_size, vocab_size)), seq_lens),
            # Wrong seq_lens shape
            lambda: decoder.decode(
                correct_logits, torch.full((batch_size, 1), seq_len, dtype=torch.int32)
            ),
        ]

        for i, error_scenario in enumerate(error_scenarios):
            try:
                error_scenario()
                pytest.fail(f"Error scenario {i+1} should have failed")
            except (ValueError, AssertionError):
                pass  # Expected error

            # Decoder should still work after each error
            labels, timesteps, seq_pos = decoder.decode(correct_logits, seq_lens)
            assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        print("Decoder maintained stability through multiple error conditions")

    def test_realistic_vocabulary_handling(self):
        """Test decoder with a realistic vocabulary structure."""
        # Create vocabulary similar to real ASR models
        vocab = [""]  # blank token

        # Add single characters
        for char in "abcdefghijklmnopqrstuvwxyz":
            vocab.append(char)

        # Add common punctuation and symbols
        for symbol in ["'", " ", ".", ",", "?", "!"]:
            vocab.append(symbol)

        # Add BPE-style subwords
        common_prefixes = ["##un", "##re", "##in", "##dis"]
        common_suffixes = ["##ing", "##ed", "##er", "##est", "##ly", "##ion", "##tion"]

        vocab.extend(common_prefixes)
        vocab.extend(common_suffixes)

        # Add some full words
        common_words = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our"]
        vocab.extend(common_words)

        vocab_size = len(vocab)

        # Create decoder with this vocabulary
        params = {
            "thread_count": 4,
            "blank_id": 0,
            "cutoff_top_n": 50,
            "cutoff_prob": 1.0,
            "alpha": 0.15,
            "beta": 0.25,
            "beam_width": 50,
            "unk_lexicon_penalty": -8.0,
            "min_tok_prob": -6.0,
            "max_beam_deviation": -12.0,
            "tok_sep": "#",
            "lm_path": None,
            "lexicon_fst_path": None,
        }

        decoder = CTCBeamDecoder(vocab=vocab, **params)

        # Test with this realistic vocabulary
        batch_size = 4
        seq_len = 300

        logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

        assert labels.shape == (batch_size, decoder.beam_width, seq_len)

        # Check that we get reasonable token diversity in outputs
        for b in range(batch_size):
            for beam in range(min(3, decoder.beam_width)):  # Check first few beams
                start_idx = seq_pos[b, beam].item()
                if start_idx > 0:
                    beam_labels = labels[b, beam, start_idx:]
                    unique_tokens = torch.unique(
                        beam_labels[beam_labels > 0]
                    )  # Non-blank tokens

                    # Should have some variety of tokens
                    assert (
                        len(unique_tokens) > 0
                    ), f"No non-blank tokens in batch {b}, beam {beam}"

        print(f"Successfully tested with realistic vocabulary of size {vocab_size}")


class TestCTCBeamDecoderRealWorldScenarios:
    """Test scenarios that mimic real-world usage patterns."""

    def test_streaming_like_processing(self, realistic_decoder):
        """Simulate streaming-like processing with overlapping segments."""
        segment_length = 200
        overlap = 50
        total_length = 1000
        vocab_size = realistic_decoder.vocab_size

        # Generate full sequence
        full_logits = torch.randn((1, total_length, vocab_size)).softmax(dim=2)

        results = []

        # Process in overlapping segments
        for start in range(
            0, total_length - segment_length + 1, segment_length - overlap
        ):
            end = start + segment_length
            segment_logits = full_logits[:, start:end, :]
            seq_lens = torch.full((1,), segment_length, dtype=torch.int32)

            labels, timesteps, seq_pos = realistic_decoder.decode(
                segment_logits, seq_lens
            )

            results.append(
                {
                    "start": start,
                    "end": end,
                    "labels": labels,
                    "timesteps": timesteps,
                    "seq_pos": seq_pos,
                }
            )

        # All segments should process successfully
        assert len(results) > 0

        for i, result in enumerate(results):
            assert result["labels"].shape == (
                1,
                realistic_decoder.beam_width,
                segment_length,
            )
            print(
                f"Segment {i+1}: [{result['start']}:{result['end']}] processed successfully"
            )

    def test_variable_quality_inputs(self, realistic_decoder):
        """Test with inputs of varying quality/confidence."""
        batch_size = 4
        seq_len = 200
        vocab_size = realistic_decoder.vocab_size

        # Create inputs with different confidence levels
        confidence_levels = [0.1, 0.3, 0.7, 0.9]  # Low to high confidence

        for i, confidence in enumerate(confidence_levels):
            # Generate logits with different entropy levels
            if confidence > 0.5:
                # High confidence: lower entropy (more peaked distribution)
                logits = torch.randn((1, seq_len, vocab_size)) * (2.0 - confidence)
            else:
                # Low confidence: higher entropy (more uniform distribution)
                logits = torch.randn((1, seq_len, vocab_size)) * (1.0 / confidence)

            logits = logits.softmax(dim=2)
            seq_lens = torch.full((1,), seq_len, dtype=torch.int32)

            labels, timesteps, seq_pos = realistic_decoder.decode(logits, seq_lens)

            assert labels.shape == (1, realistic_decoder.beam_width, seq_len)

            print(f"Processed input with confidence level {confidence:.1f}")

    def test_mixed_content_types(self, realistic_decoder):
        """Test with mixed content (speech, silence, noise simulation)."""
        batch_size = 3  # Different content types
        seq_len = 300
        vocab_size = realistic_decoder.vocab_size

        # Batch 0: Simulate speech (structured patterns)
        speech_logits = torch.randn((1, seq_len, vocab_size))
        # Add structure to simulate actual speech patterns
        for t in range(0, seq_len, 20):
            if t + 10 < seq_len:
                # Boost non-blank tokens in chunks
                speech_logits[0, t : t + 10, 1 : min(20, vocab_size)] += 0.8

        # Batch 1: Simulate silence (mostly blank)
        silence_logits = torch.randn((1, seq_len, vocab_size))
        silence_logits[0, :, 0] += 2.0  # Strong bias toward blank

        # Batch 2: Simulate noise (very random)
        noise_logits = (
            torch.randn((1, seq_len, vocab_size)) * 0.5
        )  # Lower magnitude, more random

        # Combine all content types
        combined_logits = torch.cat(
            [speech_logits, silence_logits, noise_logits], dim=0
        )
        combined_logits = combined_logits.softmax(dim=2)

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32)

        labels, timesteps, seq_pos = realistic_decoder.decode(combined_logits, seq_lens)

        assert labels.shape == (batch_size, realistic_decoder.beam_width, seq_len)

        # Analyze results for different content types
        for b in range(batch_size):
            content_type = ["speech", "silence", "noise"][b]

            # Look at best beam
            best_beam_labels = labels[b, 0, :]
            non_blank_ratio = (best_beam_labels != 0).float().mean().item()

            print(
                f"{content_type.capitalize()} content: {non_blank_ratio:.3f} non-blank ratio"
            )

            # Silence should have fewer non-blank tokens
            if content_type == "silence":
                assert (
                    non_blank_ratio < 0.5
                ), f"Silence should have low non-blank ratio, got {non_blank_ratio:.3f}"
