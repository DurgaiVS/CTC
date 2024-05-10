#include "zctc/decoder.hh"
#include <vector>
#include <iostream>


void display_help() {

    std::cout << "Usage:" << std::endl;
    std::cout << "\t ./zctc lm_path lexicon_path vocab_path timesteps" << std::endl;

    std::cout << "lm_path - Language Model (KenLM built) file path " << std::endl;
    std::cout << "lexicon_path - Lexicon fst file path " << std::endl;
    std::cout << "vocab_path - Vocabulary file path " << std::endl;
    std::cout << "timesteps - Number of timesteps to generate random samples of logits" << std::endl;

}

int load_vocab(std::vector<std::string>& vocab, char* vocab_path) {

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


int main(int argc, char** argv) {

    if (argc != 5) {
        display_help();
        return 1;
    }

    char tok_sep = '#';
    int thread_count = 1;
    int blank_id = 0;
    int cutoff_top_n = 25;
    int ts = std::stoi(argv[4]);
    float nucleus_prob_per_timestep = 1.0;
    float penalty = -5.0;
    float lm_alpha = 0.017;
    std::size_t beam_width = 20;
    char* lm_path = argv[1];
    char* lexicon_path = argv[2];
    char* vocab_path = argv[3];
    std::vector<std::string> vocab;
    int apostrophe_id = load_vocab(vocab, vocab_path);

    zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, apostrophe_id, nucleus_prob_per_timestep, lm_alpha, beam_width, penalty, tok_sep, vocab, lm_path, lexicon_path);

    std::vector<float> logits(decoder.vocab_size * ts, 1);
    std::vector<int> sorted_indices(decoder.vocab_size * ts, 1);
    std::vector<int> labels(decoder.beam_width, 0);
    std::vector<int> timesteps(decoder.beam_width, 0);

    zctc::decode<float>(&decoder, logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), 2);

    return 0;
}
