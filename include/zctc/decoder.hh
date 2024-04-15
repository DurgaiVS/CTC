#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <algorithm>
#include <string>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "./trie.hh"
#include "./ext_scorer.hh"

namespace py = pybind11;

namespace zctc {

class Decoder {
public:
    template <typename T>
    static bool descending_compare(zctc::Node<T>* x, zctc::Node<T>* y);

    const char tok_sep;
    const int thread_count, blank_id, cutoff_top_n;
    const float nucleus_prob_per_timestep, penalty;
    const std::size_t beam_width;

    int vocab_size, apostrophe_id;
    ExternalScorer ext_scorer;
    std::vector<std::string> id_tok_map;

    Decoder(int thread_count, int blank_id, int cutoff_top_n, float nucleus_prob_per_timestep, std::size_t beam_width, float penalty, char tok_sep, char* lm_path, char* lexicon_path, char* vocab_path)
        : tok_sep(tok_sep),
          thread_count(thread_count),
          blank_id(blank_id),
          cutoff_top_n(cutoff_top_n),
          nucleus_prob_per_timestep(nucleus_prob_per_timestep),
          penalty(penalty),
          beam_width(beam_width),
          ext_scorer(zctc::ExternalScorer::construct_class(lm_path, lexicon_path))
    {
        this->load_vocab(vocab_path);
    }

    inline std::string get_token_from_id(int id) const;
    void load_vocab(char* vocab_path);

    template <typename T>
    inline void start_of_word_check(Node<T>* prefix) const;

    template <typename T>
    void decode(
        T* log_logits,
        int* sorted_ids,
        int* labels,
        int* timesteps,
        const int seq_len
    ) const;

    template <typename T>
    void batch_decode(
        py::array_t<T>& batch_log_logits,
        py::array_t<int>& batch_sorted_ids,
        py::array_t<int>& batch_labels,
        py::array_t<int>& batch_timesteps,
        const int batch_size,
        const int seq_len
    ) const {
        py::buffer_info logits_buf = batch_log_logits.request();
        py::buffer_info ids_buf = batch_sorted_ids.request();
        py::buffer_info labels_buf = batch_labels.request();
        py::buffer_info timesteps_buf = batch_timesteps.request();

        if (logits_buf.ndim != 3 || ids_buf.ndim != 3 || labels_buf.ndim != 3 || timesteps_buf.ndim != 3)
            throw std::runtime_error("Number of dimensions must be three");

        int *ids = (int*)ids_buf.ptr, *labels = (int*)labels_buf.ptr, *timesteps = (int*)timesteps_buf.ptr;
        T* logits = (T*)logits_buf.ptr;

        for (int i = 0, ip_pos = 0, op_pos = 0; i < batch_size; i++) {
            ip_pos = i * seq_len * this->vocab_size;
            op_pos = i * this->beam_width * seq_len;
            this->decode(logits + ip_pos, ids + ip_pos, labels + op_pos, timesteps + op_pos, seq_len);
        }

    }

};

} // namespace zctc

/* ---------------------------------------------------------------------------- */


template <typename T>
bool zctc::Decoder::descending_compare(zctc::Node<T>* x, zctc::Node<T>* y) {
    return x->score > y->score;
}

std::string zctc::Decoder::get_token_from_id(int id) const {
    return this->id_tok_map[id];
}

void zctc::Decoder::load_vocab(char* vocab_path) {

    std::ifstream inputFile(vocab_path);

    if (!inputFile.is_open())
        std::runtime_error("Cannot open vocab file from the path provided.");

    std::string line;
    while (std::getline(inputFile, line)) {
        this->id_tok_map.push_back(line);
    }

    this->vocab_size = this->id_tok_map.size();

}

template <typename T>
void zctc::Decoder::start_of_word_check(Node<T>* prefix) const {
    prefix->is_start_of_word = !(
        prefix->id == this->apostrophe_id || 
        prefix->parent->id == this->apostrophe_id ||
        prefix->token.at(0) == this->tok_sep
    );

    if (prefix->is_start_of_word)
        prefix->lexicon_state = this->ext_scorer.lexicon->Start();

}

template <typename T>
void zctc::Decoder::decode(
    T* logits,
    int* ids,
    int* label,
    int* timestep,
    const int seq_len
) const {

    T nucleus_max = static_cast<T>(this->nucleus_prob_per_timestep);

    bool is_repeat;
    int iter_val;
    int *curr_id, *curr_l, *curr_t;
    zctc::Node<T>* child;
    std::vector<zctc::Node<T>*> prefixes, tmp;
    zctc::Node<T> root(zctc::ROOT_ID, -1, static_cast<T>(zctc::ZERO), static_cast<T>(this->penalty), "<s>", nullptr);
    fst::SortedMatcher<fst::StdVectorFst> matcher(this->ext_scorer.lexicon, fst::MATCH_INPUT);
    root.lexicon_state = this->ext_scorer.lexicon->Start();
    root.lm_state = this->ext_scorer.lm->BeginSentenceState();

    prefixes.reserve(this->beam_width);
    tmp.reserve(this->beam_width);

    prefixes.push_back(&root);


    for (int t = 0; t < seq_len; t++) {
        T nucleus_count = 0;
        iter_val = t * this->vocab_size;
        curr_id = ids + iter_val;

        for (int i = 0; i < this->cutoff_top_n; i++, curr_id++) {
            int index = *curr_id;
            T prob = logits[iter_val + index];

            nucleus_count += prob;

            for (zctc::Node<T>* prefix : prefixes) {

                child = prefix->add_to_child(index, t, prob, this->get_token_from_id(index), &is_repeat);
                tmp.push_back(child);

                if (is_repeat) 
                    continue;
                else if (child->id == this->blank_id) {
                    child->lm_state = child->parent->lm_state;
                    child->lexicon_state = child->parent->lexicon_state;

                    continue;
                }

                this->start_of_word_check(child);
                this->ext_scorer.run_ext_scoring(child, &matcher);

            }

            if (nucleus_count >= nucleus_max) break;
        }

        prefixes.clear();

        if (tmp.size() < this->beam_width) {
            std::copy(tmp.begin(), tmp.end(), std::back_inserter(prefixes));

            tmp.clear();
            continue;
        }

        std::nth_element(tmp.begin(), tmp.begin() + this->beam_width, tmp.end(), Decoder::descending_compare<T>);

        std::copy_n(tmp.begin(), this->beam_width, std::back_inserter(prefixes));
        tmp.clear();

    }

    std::sort(prefixes.begin(), prefixes.end(), Decoder::descending_compare<T>);

    iter_val = 1;
    for (zctc::Node<T>* prefix : prefixes) {
        curr_t = timestep + ((seq_len * iter_val) - 1);
        curr_l = label + ((seq_len * iter_val) - 1);

        while (prefix->parent != nullptr) {
            *curr_l = prefix->id;
            *curr_t = prefix->timestep;

            // if index becomes less than 0, might throw error
            curr_l--;
            curr_t--;

            prefix = prefix->parent;
        }

        iter_val++;
    }

}

#endif // _ZCTC_DECODER_H
