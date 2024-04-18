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
    std::size_t beam_width = 20;
    char* lm_path = argv[1];
    char* lexicon_path = argv[2];
    char* vocab_path = argv[3];

    zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, nucleus_prob_per_timestep, beam_width, penalty, tok_sep, lm_path, lexicon_path, vocab_path);

    std::vector<float> logits(decoder.vocab_size * ts, 1);
    std::vector<int> sorted_indices(decoder.vocab_size * ts, 1);
    std::vector<int> labels(decoder.beam_width, 0);
    std::vector<int> timesteps(decoder.beam_width, 0);

    decoder.decode(logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), 2);

    return 0;
}
