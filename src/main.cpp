#include "zctc/decoder.hh"
#include <vector>
#include <iostream>


void display_help() {

    std::cout << "Usage:" << std::endl;
    std::cout << "\t ./zctc thread_count blank_id cutoff_top_n nucleus_prob_per_timestep beam_width penalty tok_sep lm_path lexicon_path vocab_path timesteps" << std::endl;
    std::cout << "thread_count - To add parallelism (NOT YET INCLUDED) " << std::endl;
    std::cout << "blank_id - ID number of blank token from the vocab " << std::endl;
    std::cout << "cutoff_top_n - Number of max tokens to take from a timestep to expand the beam search " << std::endl;
    std::cout << "nucleus_prob_per_timestep - Tokens contributed to reach this probability count will be taken from a timestep" << std::endl;
    std::cout << "beam_width - Number of beams to finilize after parsing a timestep " << std::endl;
    std::cout << "penalty - Penalty number in float for OOV words " << std::endl;
    std::cout << "tok_sep - Token seperator character used by the tokenizer " << std::endl;
    std::cout << "lm_path - Language Model (KenLM built) file path " << std::endl;
    std::cout << "lexicon_path - Lexicon fst file path " << std::endl;
    std::cout << "vocab_path - Vocabulary file path " << std::endl; 

}


int main(int argc, char** argv) {

    if (argc != 12) {
        display_help();
        return 1;
    }

    char tok_sep = *(argv[7]);
    int thread_count = std::stoi(argv[1]); 
    int blank_id = std::stoi(argv[2]); 
    int cutoff_top_n = std::stoi(argv[3]);
    int ts = std::stoi(argv[11]);
    float nucleus_prob_per_timestep = std::stof(argv[4]);
    float penalty = std::stoi(argv[6]);
    std::size_t beam_width = std::stoi(argv[5]);
    char* lm_path = argv[8];
    char* lexicon_path = argv[9];
    char* vocab_path = argv[10];

    zctc::Decoder decoder(thread_count, blank_id, cutoff_top_n, nucleus_prob_per_timestep, beam_width, penalty, tok_sep, lm_path, lexicon_path, vocab_path);

    std::vector<float> logits(decoder.vocab_size * ts, 1);
    std::vector<int> sorted_indices(decoder.vocab_size * ts, 1);
    std::vector<int> labels(decoder.beam_width, 0);
    std::vector<int> timesteps(decoder.beam_width, 0);

    decoder.decode(logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), 2);

    return 0;
}
