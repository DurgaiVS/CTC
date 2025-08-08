"""
Performance tests for CTCBeamDecoder comparing different implementations.
This mirrors the performance testing done in the main test.py file.
"""

import gc
import time
from typing import Any, Dict

import pytest
import torch

from zctc import CTCBeamDecoder


class TestCTCBeamDecoderPerformanceComparison:
    """Performance comparison tests for CTCBeamDecoder implementations."""

    @pytest.fixture
    def performance_setup(self, sample_vocab, decoder_params):
        """Setup for performance testing similar to the main test.py."""
#Parameters from test.py
        config = {
            "batch_size": 4,
            "seq_len": 750,
            "beam_width": 25,
            "cutoff_top_n": 20,
            "cutoff_prob": 1.0,
            "blank_id": 0,
            "alpha": 0.17,
            "beta": 0.24,
            "min_tok_prob": -5.0,
            "max_beam_deviation": -10.0,
            "unk_lexicon_penalty": -5.0,
            "tok_sep": "#",
        }

#Update decoder params with test.py values
        params = decoder_params.copy()
        params.update(
            {
                "thread_count": config["batch_size"],
                "beam_width": config["beam_width"],
                "cutoff_top_n": config["cutoff_top_n"],
                "cutoff_prob": config["cutoff_prob"],
                "blank_id": config["blank_id"],
                "alpha": config["alpha"],
                "beta": config["beta"],
                "min_tok_prob": config["min_tok_prob"],
                "max_beam_deviation": config["max_beam_deviation"],
                "unk_lexicon_penalty": config["unk_lexicon_penalty"],
                "tok_sep": config["tok_sep"],
            }
        )

        decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

        return {"decoder": decoder, "config": config, "vocab_size": len(sample_vocab)}

    def test_zctc_performance_benchmark(self, performance_setup):
        """Benchmark ZCTC performance similar to test.py implementation."""
        decoder = performance_setup["decoder"]
        config = performance_setup["config"]
        vocab_size = performance_setup["vocab_size"]

        batch_size = config["batch_size"]
        seq_len = config["seq_len"]

        seq_lens = torch.empty((batch_size,), dtype=torch.int32).fill_(seq_len)

#Warmup runs
        warmup_iterations = 3
        for _ in range(warmup_iterations):
            logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)
            logits = logits.softmax(dim=2)  # Convert to probabilities

            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

#Clear memory
            del labels, timesteps, seq_pos, logits
            gc.collect()

#Actual performance test
        test_iterations = 50
        total_time = 0.0

        for i in range(test_iterations):
            logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.float32)
            logits = logits.softmax(dim=2)  # Convert to probabilities

            start_time = time.time()
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            end_time = time.time()

            total_time += end_time - start_time

#Cleanup
            del labels, timesteps, seq_pos, logits

            if (i + 1) % 10 == 0:
                current_avg = total_time / (i + 1)
                print(f"Completed {i+1} iterations, avg time: {current_avg:.5f}s")

        avg_time = total_time / test_iterations
        print(f"\nZCTC Decoder Performance:")
        print(f"Average time per batch: {avg_time:.5f} seconds")
        print(f"Batch size: {batch_size}, Seq length: {seq_len}")
        print(f"Vocab size: {vocab_size}, Beam width: {config['beam_width']}")

#Performance assertions
        assert avg_time < 1.0, f"ZCTC decoder too slow: {avg_time:.5f}s per batch"
        assert total_time < 30.0, f"Total test time too long: {total_time:.2f}s"

        return avg_time

    def test_batch_size_scaling(self, sample_vocab, decoder_params):
        """Test how performance scales with batch size."""
        batch_sizes = [1, 2, 4, 8, 16]
        seq_len = 200
        vocab_size = len(sample_vocab)

        results = {}

        for batch_size in batch_sizes:
            params = decoder_params.copy()
            params["thread_count"] = batch_size
            decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

            seq_lens = torch.empty((batch_size,), dtype=torch.int32).fill_(seq_len)

#Warmup
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            decoder.decode(logits, seq_lens)

#Performance test
            iterations = 10
            total_time = 0.0

            for _ in range(iterations):
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)

                start_time = time.time()
                labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
                end_time = time.time()

                total_time += end_time - start_time

            avg_time = total_time / iterations
            avg_time_per_sample = avg_time / batch_size

            results[batch_size] = {
                "avg_time": avg_time,
                "avg_time_per_sample": avg_time_per_sample,
            }

            print(
                f"Batch size {batch_size}: {avg_time:.5f}s total, {avg_time_per_sample:.5f}s per sample"
            )

#Check that scaling is reasonable
#Per - sample time shouldn't increase dramatically with batch size
        for batch_size in batch_sizes[1:]:  # Skip first element
            prev_batch_size = batch_sizes[batch_sizes.index(batch_size) - 1]
            current_per_sample = results[batch_size]["avg_time_per_sample"]
            prev_per_sample = results[prev_batch_size]["avg_time_per_sample"]

#Allow up to 50 % increase in per - sample time(this is flexible)
            max_allowed_increase = prev_per_sample * 1.5
            assert (
                current_per_sample <= max_allowed_increase
            ), f"Per-sample time increased too much from batch {prev_batch_size} to {batch_size}"

    def test_sequence_length_scaling(self, sample_vocab, decoder_params):
        """Test how performance scales with sequence length."""
        seq_lengths = [50, 100, 200, 400, 800]
        batch_size = 4
        vocab_size = len(sample_vocab)

        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        results = {}

        for seq_len in seq_lengths:
            seq_lens = torch.empty((batch_size,), dtype=torch.int32).fill_(seq_len)

#Warmup
            logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)
            decoder.decode(logits, seq_lens)

#Performance test
            iterations = 10
            total_time = 0.0

            for _ in range(iterations):
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)

                start_time = time.time()
                labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
                end_time = time.time()

                total_time += end_time - start_time

            avg_time = total_time / iterations
            avg_time_per_timestep = avg_time / (batch_size * seq_len)

            results[seq_len] = {
                "avg_time": avg_time,
                "avg_time_per_timestep": avg_time_per_timestep,
            }

            print(
                f"Seq length {seq_len}: {avg_time:.5f}s total, {avg_time_per_timestep:.8f}s per timestep"
            )

#Check that scaling is reasonable(roughly linear with sequence length)
        for seq_len in seq_lengths[1:]:
            prev_seq_len = seq_lengths[seq_lengths.index(seq_len) - 1]
            current_time = results[seq_len]["avg_time"]
            prev_time = results[prev_seq_len]["avg_time"]

#Time should scale roughly linearly with sequence length
            expected_ratio = seq_len / prev_seq_len
            actual_ratio = current_time / prev_time

#Allow up to 50 % deviation from linear scaling
            assert (
                actual_ratio <= expected_ratio * 1.5
            ), f"Performance scaling worse than expected from {prev_seq_len} to {seq_len}"

    def test_memory_efficiency(self, performance_setup):
        """Test memory usage during decoding."""
        decoder = performance_setup["decoder"]
        config = performance_setup["config"]
        vocab_size = performance_setup["vocab_size"]

        batch_size = config["batch_size"]
        seq_len = config["seq_len"]

#Test with larger batches to see memory usage
        large_batch_size = batch_size * 4
        seq_lens = torch.empty((large_batch_size,), dtype=torch.int32).fill_(seq_len)

try :
#This should work without running out of memory
	logits = torch.randn((large_batch_size, seq_len, vocab_size)).softmax(dim=2)
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)

#Check that we got reasonable output
            assert labels.shape[0] == large_batch_size
            assert timesteps.shape[0] == large_batch_size
            assert seq_pos.shape[0] == large_batch_size

            print(f"Successfully processed large batch of size {large_batch_size}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping large batch test due to memory constraints: {e}")
            else:
                raise

    def test_concurrent_decoding(self, sample_vocab, decoder_params):
        """Test that multiple decoder instances can work concurrently."""
        import queue
        import threading

        num_decoders = 4
        batch_size = 2
        seq_len = 100
        vocab_size = len(sample_vocab)

#Create multiple decoder instances
        decoders = []
        for i in range(num_decoders):
            decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
            decoders.append(decoder)

        results_queue = queue.Queue()

        def decode_worker(decoder_id, decoder):
            """Worker function for concurrent decoding."""
	try : seq_lens = torch.empty((batch_size,), dtype=torch.int32).fill_(seq_len)
                logits = torch.randn((batch_size, seq_len, vocab_size)).softmax(dim=2)

                start_time = time.time()
                labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
                end_time = time.time()

                results_queue.put(
                    {
                        "decoder_id": decoder_id,
                        "success": True,
                        "time": end_time - start_time,
                        "shapes": (labels.shape, timesteps.shape, seq_pos.shape),
                    }
                )
            except Exception as e:
                results_queue.put(
                    {"decoder_id": decoder_id, "success": False, "error": str(e)}
                )

#Start all threads
        threads = []
        for i, decoder in enumerate(decoders):
            thread = threading.Thread(target=decode_worker, args=(i, decoder))
            threads.append(thread)
            thread.start()

#Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                pytest.fail("Thread did not complete within timeout")

#Check results
        results = []
        for _ in range(num_decoders):
            result = results_queue.get_nowait()
            results.append(result)

#All decoders should have succeeded
        for result in results:
            assert result[
                "success"
            ], f"Decoder {result['decoder_id']} failed: {result.get('error', 'Unknown error')}"
            print(f"Decoder {result['decoder_id']}: {result['time']:.5f}s")

#All results should have the same shapes
        expected_shapes = results[0]["shapes"]
        for result in results[1:]:
            assert (
                result["shapes"] == expected_shapes
            ), "Shape mismatch between concurrent decoders"


@pytest.mark.slow
class TestCTCBeamDecoderStressTest:
    """Stress tests for CTCBeamDecoder with large inputs."""

    def test_large_batch_stress(self, sample_vocab, decoder_params):
        """Stress test with large batch size."""
        large_batch_size = 64
        seq_len = 200
        vocab_size = len(sample_vocab)

        params = decoder_params.copy()
        params["thread_count"] = min(large_batch_size, 16)  # Reasonable thread limit
        decoder = CTCBeamDecoder(vocab=sample_vocab, **params)

        seq_lens = torch.empty((large_batch_size,), dtype=torch.int32).fill_(seq_len)

        try:
            logits = torch.randn((large_batch_size, seq_len, vocab_size)).softmax(dim=2)

            start_time = time.time()
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            end_time = time.time()

            decode_time = end_time - start_time

            assert labels.shape == (large_batch_size, decoder.beam_width, seq_len)
            assert (
                decode_time < 10.0
            ), f"Large batch decoding took too long: {decode_time:.2f}s"

            print(f"Large batch ({large_batch_size}) decoded in {decode_time:.3f}s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping large batch stress test due to memory: {e}")
            else:
                raise

    def test_long_sequence_stress(self, sample_vocab, decoder_params):
        """Stress test with very long sequences."""
        batch_size = 4
        long_seq_len = 2000
        vocab_size = len(sample_vocab)

        decoder = CTCBeamDecoder(vocab=sample_vocab, **decoder_params)
        seq_lens = torch.empty((batch_size,), dtype=torch.int32).fill_(long_seq_len)

        try:
            logits = torch.randn((batch_size, long_seq_len, vocab_size)).softmax(dim=2)

            start_time = time.time()
            labels, timesteps, seq_pos = decoder.decode(logits, seq_lens)
            end_time = time.time()

            decode_time = end_time - start_time

            assert labels.shape == (batch_size, decoder.beam_width, long_seq_len)
            assert (
                decode_time < 15.0
            ), f"Long sequence decoding took too long: {decode_time:.2f}s"

            print(f"Long sequences ({long_seq_len}) decoded in {decode_time:.3f}s")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Skipping long sequence stress test due to memory: {e}")
            else:
                raise
