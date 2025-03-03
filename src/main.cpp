#include <ctime>
#include <iostream>
#include <numeric>

#include "./defaults.hh"
#include "zctc/decoder.hh"

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

int
debug_decoder()
{

	char tok_sep = '#';
	int iter_count, blank_id, seq_len = 1000, thread_count = 1, cutoff_top_n = 40;
	float nucleus_prob_per_timestep = 1.0, penalty = -5.0, alpha = 0.017, beta = 0;
	float min_tok_prob = -5.0, max_beam_deviation = -10.0;
	std::size_t beam_width = 250;
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

	int apostrophe_id = load_vocab(vocab, vocab_path.c_str());

	zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, alpha, beta,
						  beam_width, penalty, min_tok_prob, max_beam_deviation, tok_sep, vocab, lm_path.data(),
						  lexicon_path.data());

	std::vector<float> logits(decoder.vocab_size * seq_len);
	std::vector<int> sorted_indices(decoder.vocab_size * seq_len);
	std::vector<int> labels(decoder.beam_width * seq_len, 0);
	std::vector<int> timesteps(decoder.beam_width * seq_len, 0);
	std::vector<int> seq_pos(decoder.beam_width, 0);

	fst::StdVectorFst hotwords_fst;
	std::vector<std::vector<int>> hotwords({ { 1, 2, 3, 4, 5 } });
	std::vector<float> hotwords_weight(5.0);
	zctc::populate_hotword_fst(&hotwords_fst, hotwords, hotwords_weight);

	// To generate random values for logits
	std::random_device rnd_device;
	std::mt19937 mersenne_engine { rnd_device() };
	std::uniform_real_distribution<float> dist { 0.0f, 0.05f };
	auto gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };

	for (int t = 0, temp = 0; t < iter_count; t++) {

		std::generate(logits.begin(), logits.end(), gen);

		// To get the sorted indices for the logits, timesteps wise
		for (int i = 0; i < seq_len; i++) {
			temp = i * decoder.vocab_size;
			std::iota(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size), 0);
			std::stable_sort(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size),
							 [&logits, &temp](int a, int b) { return logits[temp + a] > logits[temp + b]; });
		}

		zctc::decode<float>(&decoder, logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), seq_len,
							seq_len, seq_pos.data(), &hotwords_fst);

		std::fill(labels.begin(), labels.end(), 0);
		std::fill(timesteps.begin(), timesteps.end(), 0);
		std::fill(seq_pos.begin(), seq_pos.end(), 0);
	}

	return 0;
}

int
debug_decoder_with_constants()
{

	char tok_sep = '#';
	int blank_id = 0, thread_count = 1, cutoff_top_n = 40;
	float nucleus_prob_per_timestep = 1.0, penalty = -5.0, alpha = 0.17, beta = 0;
	float min_tok_prob = -5.0, max_beam_deviation = -10.0;

	std::size_t beam_width = 281;
	std::string vocab_path;
	std::vector<std::string> vocab;

	std::cout << "Enter vocab path: ";
	std::cin >> vocab_path;
	std::cout << "Enter blank id: ";
	std::cin >> blank_id;

	int apostrophe_id = load_vocab(vocab, vocab_path.c_str());

	zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, alpha, beta,
						  beam_width, penalty, min_tok_prob, max_beam_deviation, tok_sep, vocab, nullptr, nullptr);

	std::vector<float>& logits = defaults::logits;
	std::vector<int> sorted_indices(decoder.vocab_size * defaults::seq_len);
	std::vector<int> labels(decoder.beam_width * defaults::seq_len, 0);
	std::vector<int> timesteps(decoder.beam_width * defaults::seq_len, 0);
	std::vector<int> seq_pos(decoder.beam_width, 0);

	// To get the sorted indices for the logits, timesteps wise
	for (int i = 0, temp = 0; i < defaults::seq_len; i++) {
		temp = i * decoder.vocab_size;
		std::iota(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size), 0);
		std::stable_sort(sorted_indices.begin() + temp, sorted_indices.begin() + (temp + decoder.vocab_size),
						 [&logits, &temp](int a, int b) { return logits[temp + a] > logits[temp + b]; });
	}

	for (int i = 0, j = 0; i < defaults::seq_len; i++) {
		j = i * decoder.vocab_size;
		if (sorted_indices[j] == 0)
			continue;
		std::cout << i << " : " << sorted_indices[j] << ", " << sorted_indices[j + 1] << ", " << sorted_indices[j + 2]
				  << std::endl;
	}

	zctc::decode<float>(&decoder, logits.data(), sorted_indices.data(), labels.data(), timesteps.data(),
						defaults::seq_len, defaults::seq_len, seq_pos.data(), nullptr);

	/*
	NOTE: Only printing the first beam labels.
	*/
	for (int j = seq_pos[0]; j < defaults::seq_len; j++) {
		std::cout << labels[j] << ":" << timesteps[j] << " ";
	}
}

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
	std::cout << "Enter choice(0 for Decoder(with rand inputs), 1 for Decoder(with constant input), 2 for Decoder(with "
				 "toy exp), 3 for FST: ";
	std::cin >> choice;
	if (choice == 0)
		return debug_decoder();
	else if (choice == 1)
		return debug_decoder_with_constants();
	else if (choice == 2)
		return debug_decoder_with_toy_exp();
	else if (choice == 3)
		return debug_fst();
	else {
		std::cout << "Invalid choice. Exiting..." << std::endl;
		return 1;
	}
}
