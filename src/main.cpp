#include "zctc/decoder.hh"
#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>

void
display_help()
{

    std::cout << "Usage:" << std::endl;
    std::cout << "\t./zctc lm_path lexicon_path vocab_path iter_count blank_id" << std::endl;

    std::cout << "lm_path - Language Model (KenLM built) file path " << std::endl;
    std::cout << "lexicon_path - Lexicon fst file path " << std::endl;
    std::cout << "vocab_path - Vocabulary file path " << std::endl;
    std::cout << "iter_count - Number of iterations to run the decoder for with randomly generated logits" << std::endl;
    std::cout << "blank_id [optional] - Blank token ID. Default: 0" << std::endl;
}

int
load_vocab(std::vector<std::string>& vocab, char* vocab_path)
{

    std::ifstream inputFile(vocab_path);
    int apostrophe_id = -1;

    if (!inputFile.is_open())
        std::runtime_error("Cannot open vocab file from the path provided.");

    std::string line;
    while (std::getline(inputFile, line)) {
        vocab.push_back(line);

        if (line == "'")
            apostrophe_id = vocab.size() - 1;
    }

    return apostrophe_id;
}

int
main(int argc, char** argv)
{
    if (argc != 5 || argc != 6) {
        display_help();
        return 1;
    }

    char tok_sep = '#';
    char* lm_path = argv[1];
    char* lexicon_path = argv[2];
    char* vocab_path = argv[3];

    int seq_len = 250;
    int thread_count = 1;
    int cutoff_top_n = 25;
    int iter_count = std::stoi(argv[4]);
    int blank_id;
    if (argc == 6)
        blank_id = std::stoi(argv[5]);
    else
        blank_id = 0;

    float nucleus_prob_per_timestep = 1.0;
    float penalty = -5.0;
    float lm_alpha = 0.017;

    std::size_t beam_width = 20;
    std::vector<std::string> vocab;
    int apostrophe_id = load_vocab(vocab, vocab_path);

    zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, lm_alpha,
                          beam_width, penalty, tok_sep, vocab, lm_path, lexicon_path);

    std::vector<float> logits(decoder.vocab_size * seq_len);
    std::vector<int> sorted_indices(decoder.vocab_size * seq_len);
    std::vector<int> labels(decoder.beam_width * seq_len, 0);
    std::vector<int> timesteps(decoder.beam_width * seq_len, 0);

    // To generate random values for logits
    std::random_device rnd_device;
    std::mt19937 mersenne_engine { rnd_device() };
    std::uniform_real_distribution<float> dist { 0.0, 1.0 };
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

        zctc::decode<float>(&decoder, logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), seq_len);

        std::fill(labels.begin(), labels.end(), 0);
        std::fill(timesteps.begin(), timesteps.end(), 0);
    }

    return 0;
}
