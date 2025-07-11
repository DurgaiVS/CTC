#include <cassert>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

#include "zctc/decoder.hh"

/**
 * @brief Loads the vocabulary from the provided path to the vocab vector and returns the index of the apostrophe.
 *
 * @param vocab The vector to store the vocab.
 * @param vocab_path The path to the vocab file.
 *
 * @return The index of the apostrophe in the vocab.
 */
int
load_vocab(std::vector<std::string>& vocab, const char* vocab_path)
{

	std::ifstream inputFile(vocab_path);
	int apostrophe_id = -1;

	if (!inputFile.is_open())
		std::runtime_error("Cannot open vocab file from the path provided.");

	std::string line;
	while (std::getline(inputFile, line)) {
		vocab.emplace_back(line);

		if (line == "'")
			apostrophe_id = vocab.size() - 1;
	}

	return apostrophe_id;
}

template <typename T>
void
normalise(std::vector<T>& logits)
{
	T sum = 0, max = *std::max_element(logits.begin(), logits.end());
	for (T& i : logits) {
		i = i - max;
		sum += i;
	}
	for (T& i : logits) {
		i /= sum;
	}
}

template <typename T>
void
normalise(T* logits, int size)
{
	T sum = 0, max = std::numeric_limits<T>::lowest();
	T* temp = logits;
	for (int i = 0; i < size; i++, temp++) {
		if (*temp > max)
			max = *temp;
	}

	temp = logits;
	for (int i = 0; i < size; i++, temp++) {
		*temp = *temp - max;
		sum += *temp;
	}

	temp = logits;
	for (int i = 0; i < size; i++, temp++) {
		*temp = *temp / sum;
	}
}

/**
 * @brief Test the decoder with random logits.
 *
 * @return int 0 on successful execution
 */
int
debug_decoder()
{

	char tok_sep = '#';
	int iter_count, blank_id, seq_len = 1000, thread_count = 1, cutoff_top_n = 40;
	int batch_size = 4;
	float nucleus_prob_per_timestep = 1.0, penalty = -5.0, alpha = 0.017, beta = 0;
	float min_tok_prob = -10.0, max_beam_deviation = -20.0;
	std::size_t beam_width = 25;
	std::string lm_path, lexicon_path, vocab_path;
	std::vector<std::string> vocab;

	std::cout << "Enter lm path: ";
	std::cin >> lm_path;
	std::cout << "Enter lexicon path: ";
	std::cin >> lexicon_path;
	std::cout << "Enter vocab path: ";
	std::cin >> vocab_path;
	std::cout << "Enter number of iterations to run: ";
	std::cin >> iter_count;
	std::cout << "Enter blank id: ";
	std::cin >> blank_id;
	std::cout << "Enter alpha[" << alpha << "]: ";
	std::cin >> alpha;
	std::cout << "Enter beta[" << beta << "]: ";
	std::cin >> beta;

	int apostrophe_id = load_vocab(vocab, vocab_path.c_str());

	zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, alpha, beta,
						  beam_width, penalty, min_tok_prob, max_beam_deviation, tok_sep, vocab, lm_path.data(),
						  lexicon_path.data());

	std::vector<float> logits(batch_size * decoder.vocab_size * seq_len);
	std::vector<int> sorted_indices(batch_size * decoder.vocab_size * seq_len);
	std::vector<int> labels(batch_size * decoder.beam_width * seq_len, 0);
	std::vector<int> timesteps(batch_size * decoder.beam_width * seq_len, 0);
	std::vector<int> seq_lens(batch_size, seq_len);
	std::vector<int> seq_pos(batch_size * decoder.beam_width, 0);

	std::vector<std::vector<int>> hotwords({ { 1, 2, 3, 4, 5 }, { 1, 5, 7, 9, 11 }, { 3, 6, 9 } });
	std::vector<float> hotwords_weight({ 5.0, 10.0, 20.0 });

	// To generate random values for logits
	std::chrono::milliseconds duration(0);
	std::random_device rnd_device;
	std::mt19937 mersenne_engine { rnd_device() };
	// std::uniform_real_distribution<float> dist { 0.0f, 0.05f };
	std::normal_distribution<float> dist { 0.1f, 3.0f };
	auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

	for (int t = 1; t <= iter_count; t++) {

		std::cout << "\rIteration: " << t << " / " << iter_count << " [" << duration.count() << " ms / it]"
				  << std::flush;
		std::generate(logits.begin(), logits.end(), gen);

		// To get the sorted indices for the logits, timesteps wise
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0, temp = 0; j < seq_len; j++) {
				temp = (i * seq_len * decoder.vocab_size) + (j * decoder.vocab_size);
				normalise(logits.data() + temp, decoder.vocab_size);
				std::iota(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size), 0);
				std::stable_sort(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size),
								 [&logits, &temp](int a, int b) { return logits[temp + a] > logits[temp + b]; });
			}
		}

		auto start = std::chrono::high_resolution_clock::now();
		decoder.serial_decode(logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), seq_lens.data(),
							  seq_pos.data(), batch_size, seq_len, hotwords, hotwords_weight, nullptr);
		auto end = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < beam_width; j++) {
				int prev_val = -1, curr_val = -1;
				for (int k = seq_pos[(i * beam_width) + j]; k < seq_len; k++) {
					curr_val = timesteps[(i * beam_width * seq_len) + (j * seq_len) + k];
					assert(("The timestep is not in increasing order...", prev_val < curr_val));
					prev_val = curr_val;
				}
			}
		}

		std::fill(labels.begin(), labels.end(), 0);
		std::fill(timesteps.begin(), timesteps.end(), 0);
		std::fill(seq_pos.begin(), seq_pos.end(), 0);
	}

	std::cout << std::endl;

	return 0;
}

/**
 * @brief A toy experiment to test the decoder.
 *
 * @return int 0 on successful execution
 */
int
debug_decoder_with_toy_exp()
{

	char tok_sep = '#';
	int blank_id = 0, seq_len = 2, thread_count = 1, cutoff_top_n = 3;
	float nucleus_prob_per_timestep = 1.0, penalty = -5.0, alpha = 0.017, beta = 0;
	float min_tok_prob = -5.0, max_beam_deviation = -10.0;
	std::size_t beam_width = 9;
	std::vector<std::string> vocab = { "_", "'", "b" };

	zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, 1, nucleus_prob_per_timestep, alpha, beta, beam_width,
						  penalty, min_tok_prob, max_beam_deviation, tok_sep, vocab, nullptr, nullptr);

	std::vector<float> logits = { 0.6, 0.3, 0.1, 0.6, 0.35, 0.05 };
	std::vector<int> sorted_indices(decoder.vocab_size * seq_len);
	std::vector<int> labels(decoder.beam_width * seq_len, 0);
	std::vector<int> timesteps(decoder.beam_width * seq_len, 0);
	std::vector<int> seq_pos(decoder.beam_width, 0);

	// To get the sorted indices for the logits, timesteps wise
	for (int i = 0, temp = 0; i < seq_len; i++) {
		temp = i * decoder.vocab_size;
		std::iota(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size), 0);
		std::stable_sort(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size),
						 [&logits, &temp](int a, int b) { return logits[temp + a] > logits[temp + b]; });
	}

	zctc::decode<float>(&decoder, logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), seq_len,
						seq_len, seq_pos.data(), nullptr);

	for (int i = 0; i < decoder.beam_width; i++) {
		for (int j = 0; j < seq_len; j++) {
			std::cout << labels[i * seq_len + j] << ":" << timesteps[i * seq_len + j] << " ";
		}
		std::cout << std::endl;
	}
}

/**
 * @brief Test the FST with the provided vocab and lexicon file.
 *
 * @return int 0 on successful execution
 */
int
debug_fst()
{
	std::string vocab_path, file_path;

	std::cout << "Enter vocab path: ";
	std::cin >> vocab_path;
	std::cout << "Enter tokenized lexicon path: ";
	std::cin >> file_path;

	zctc::ZFST zfst(vocab_path.data(), (char*)nullptr);
	zctc::parse_lexicon_file(&zfst, file_path, 0);

	return 0;
}

int
main(int argc, char** argv)
{
	int choice;
	std::cout << "Enter choice(0 for Decoder(with rand inputs), 1 for Decoder(with toy exp), 2 for FST: ";
	std::cin >> choice;

	if (choice == 0)
		return debug_decoder();
	else if (choice == 1)
		return debug_decoder_with_toy_exp();
	else if (choice == 2)
		return debug_fst();
	else {
		std::cout << "Invalid choice. Exiting..." << std::endl;
		return 1;
	}
}
