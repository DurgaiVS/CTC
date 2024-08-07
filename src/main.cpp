#include <algorithm>
#include <ctime>
#include <iostream>
#include <numeric>
#include <vector>

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
        vocab.push_back(line);

        if (line == "'")
            apostrophe_id = vocab.size() - 1;
    }

    return apostrophe_id;
}

int
load_vocab(std::unordered_map<std::string, int>& char_map, const char* vocab_path)
{
    std::ifstream inputFile(vocab_path);
    int id = 0;

    if (!inputFile.is_open())
        std::runtime_error("Cannot open vocab file from the path provided.");

    std::string line;
    while (std::getline(inputFile, line)) {
        char_map[line] = id++;
    }

    return 0;
}

int
debug_decoder()
{

    char tok_sep = '#';
    int iter_count, blank_id, seq_len = 1000, thread_count = 1, cutoff_top_n = 40;
    float nucleus_prob_per_timestep = 1.0, penalty = -5.0, lm_alpha = 0.017;
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

    zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, lm_alpha,
                          beam_width, penalty, tok_sep, vocab, lm_path.data(), lexicon_path.data());

    std::vector<float> logits(decoder.vocab_size * seq_len);
    std::vector<int> sorted_indices(decoder.vocab_size * seq_len);
    std::vector<int> labels(decoder.beam_width * seq_len, 0);
    std::vector<int> timesteps(decoder.beam_width * seq_len, 0);

    fst::StdVectorFst hotwords_fst;
    std::vector<std::vector<int>> hotwords({ { 1, 2, 3, 4, 5 } });
    std::vector<float> hotwords_weight(5.0);
    zctc::populate_hotword_fst(&hotwords_fst, hotwords, hotwords_weight);

    // To generate random values for logits
    std::random_device rnd_device;
    std::mt19937 mersenne_engine { rnd_device() };
    std::uniform_real_distribution<float> dist { 0.0f, 0.03f };
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
                            nullptr);

        std::fill(labels.begin(), labels.end(), 0);
        std::fill(timesteps.begin(), timesteps.end(), 0);
    }

    return 0;
}

int
debug_fst()
{
    std::string vocab_path, file_path;
    std::unordered_map<std::string, int> char_map;
    zctc::ZFST zfst((char*)nullptr);

    std::cout << "Enter vocab path: ";
    std::cin >> vocab_path;
    std::cout << "Enter tokenized lexicon path: ";
    std::cin >> file_path;

    load_vocab(char_map, vocab_path.c_str());
    zctc::parse_lexicon_file(&zfst, file_path, 0, char_map);

    return 0;
}

int
main(int argc, char** argv)
{
    int choice;
    std::cout << "Enter choice(0 for Decoder, 1 for FST): ";
    std::cin >> choice;
    if (choice == 0)
        return debug_decoder();
    else
        return debug_fst();
}
