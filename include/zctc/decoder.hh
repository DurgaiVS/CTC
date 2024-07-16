#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <cmath>

#include <ThreadPool.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./ext_scorer.hh"
#include "./trie.hh"
#include "./zfst.hh"

namespace py = pybind11;

namespace zctc {

class Decoder {
public:
    template <typename T>
    static bool descending_compare(zctc::Node<T>* x, zctc::Node<T>* y);

    const int thread_count, blank_id, cutoff_top_n, vocab_size;
    const float nucleus_prob_per_timestep, penalty;
    const std::size_t beam_width;
    const std::vector<std::string> vocab;
    const ExternalScorer ext_scorer;

    Decoder(int thread_count, int blank_id, int cutoff_top_n, int apostrophe_id, float nucleus_prob_per_timestep,
            float lm_alpha, std::size_t beam_width, float penalty, char tok_sep, std::vector<std::string> vocab,
            const char* lm_path, const char* lexicon_path)
        : thread_count(thread_count)
        , blank_id(blank_id)
        , cutoff_top_n(cutoff_top_n)
        , vocab_size(vocab.size())
        , nucleus_prob_per_timestep(nucleus_prob_per_timestep)
        , penalty(penalty)
        , beam_width(beam_width)
        , vocab(vocab)
        , ext_scorer(tok_sep, apostrophe_id, lm_alpha, penalty, lm_path, lexicon_path)
    {
    }

    template <typename T>
    void batch_decode(py::array_t<T>& batch_log_logits, py::array_t<int>& batch_sorted_ids,
                      py::array_t<int>& batch_labels, py::array_t<int>& batch_timesteps,
                      py::array_t<int>& batch_seq_len, const int batch_size, const int max_seq_len,
                      std::vector<std::vector<int>>& hotwords, std::vector<float>& hotwords_weight) const;
};

template <typename T>
int
decode(const Decoder* decoder, T* logits, int* ids, int* label, int* timestep, const int seq_len,
       fst::StdVectorFst* hotwords_fst)
{

    bool is_repeat, is_blank;
    int iter_val;
    T nucleus_max, nucleus_count, prob;
    int *curr_id, *curr_l, *curr_t;
    zctc::Node<T>* child;
    std::vector<zctc::Node<T>*> prefixes, tmp;
    zctc::Node<T> root(zctc::ROOT_ID, -1, true, static_cast<T>(zctc::ZERO), static_cast<T>(decoder->penalty), "<s>",
                       nullptr);
    fst::SortedMatcher<fst::StdVectorFst> lexicon_matcher(decoder->ext_scorer.lexicon, fst::MATCH_INPUT);
    fst::SortedMatcher<fst::StdVectorFst> hotwords_matcher(hotwords_fst, fst::MATCH_INPUT);

    nucleus_max = static_cast<T>(decoder->nucleus_prob_per_timestep);
    decoder->ext_scorer.initialise_start_states(&root, hotwords_fst);
    prefixes.reserve(decoder->beam_width);
    tmp.reserve(decoder->cutoff_top_n * decoder->beam_width);

    prefixes.push_back(&root);

    for (int t = 0; t < seq_len; t++) {
        nucleus_count = 0;
        iter_val = t * decoder->vocab_size;
        curr_id = ids + iter_val;

        for (int i = 0, index = 0; i < decoder->cutoff_top_n; i++, curr_id++) {
            index = *curr_id;
            prob = logits[iter_val + index];

            is_blank = index == decoder->blank_id;
            nucleus_count += prob;
            prob = std::log(prob);

            for (zctc::Node<T>* prefix : prefixes) {

                child = prefix->add_to_child(index, t, prob, decoder->vocab[index], is_blank, &is_repeat);
                tmp.push_back(child);

                if (!(is_blank || is_repeat)) {
                    // only run ext scoring for non-duplicate and non-blank tokens.
                    decoder->ext_scorer.run_ext_scoring(child, &lexicon_matcher, hotwords_fst, &hotwords_matcher);

                } else {
                    // in case of repeated token but with less confidence, the probs
                    // will get included in the path, but we also will keep track of
                    // the most confident probs and its timestep.
                    child->lm_state = prefix->lm_state;

                    child->lexicon_state = prefix->lexicon_state;
                    child->arc_exist = prefix->arc_exist;

                    child->hotword_state = prefix->hotword_state;
                    child->hotword_length = prefix->hotword_length;
                    child->hotword_weight = prefix->hotword_weight;
                    child->is_hotpath = prefix->is_hotpath;

                    // only non-blank and repeated token's LM probs will be used.
                    if (!is_blank) {
                        child->lm_prob = prefix->lm_prob;
                    }
                }

                // update total score for the node,
                // considering probs, LM probs, OOV penalty.
                child->update_score();
            }

            if (nucleus_count >= nucleus_max)
                break;
        }

        prefixes.clear();

        if (tmp.size() < decoder->beam_width) {
            std::copy(tmp.begin(), tmp.end(), std::back_inserter(prefixes));

            tmp.clear();
            continue;
        }

        std::nth_element(tmp.begin(), tmp.begin() + decoder->beam_width, tmp.end(), Decoder::descending_compare<T>);

        std::copy_n(tmp.begin(), decoder->beam_width, std::back_inserter(prefixes));
        tmp.clear();
    }

    std::stable_sort(prefixes.begin(), prefixes.end(), Decoder::descending_compare<T>);

    iter_val = 1;
    for (zctc::Node<T>* prefix : prefixes) {

        curr_t = timestep + ((seq_len * iter_val) - 1);
        curr_l = label + ((seq_len * iter_val) - 1);

        while (prefix->id != zctc::ROOT_ID) {

            *curr_l = prefix->id;
            *curr_t = prefix->timestep;

            // if index becomes less than 0, might throw error
            curr_l--;
            curr_t--;

            prefix = prefix->parent;
        }

        iter_val++;
    }

    return 0;
}

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
bool
zctc::Decoder::descending_compare(zctc::Node<T>* x, zctc::Node<T>* y)
{
    return x->score_w_h > y->score_w_h;
}

template <typename T>
void
zctc::Decoder::batch_decode(py::array_t<T>& batch_log_logits, py::array_t<int>& batch_sorted_ids,
                            py::array_t<int>& batch_labels, py::array_t<int>& batch_timesteps,
                            py::array_t<int>& batch_seq_len, const int batch_size, const int max_seq_len,
                            std::vector<std::vector<int>>& hotwords, std::vector<float>& hotwords_weight) const
{
    ThreadPool pool(this->thread_count);
    std::vector<std::future<int>> results;

    fst::StdVectorFst hotwords_fst;
    if (!hotwords.empty()) {
        populate_hotword_fst(&hotwords_fst, hotwords, hotwords_weight);
    }

    py::buffer_info logits_buf = batch_log_logits.request();
    py::buffer_info ids_buf = batch_sorted_ids.request();
    py::buffer_info labels_buf = batch_labels.request(true);
    py::buffer_info timesteps_buf = batch_timesteps.request(true);
    py::buffer_info seq_len_buf = batch_seq_len.request();

    if (logits_buf.ndim != 3 || ids_buf.ndim != 3 || labels_buf.ndim != 3 || timesteps_buf.ndim != 3
        || seq_len_buf.ndim != 1)
        throw std::runtime_error("Logits must be three dimensional, like Batch x Seq-len x Vocab, "
                                 "and Sequence Length must be one dimensional, like Batch");

    int* ids = static_cast<int*>(ids_buf.ptr);
    int* labels = static_cast<int*>(labels_buf.ptr);
    int* timesteps = static_cast<int*>(timesteps_buf.ptr);
    int* seq_len = static_cast<int*>(seq_len_buf.ptr);
    T* logits = static_cast<T*>(logits_buf.ptr);

    for (int i = 0, ip_pos = 0, op_pos = 0; i < batch_size; i++) {
        ip_pos = i * max_seq_len * this->vocab_size;
        op_pos = i * this->beam_width * max_seq_len;

        results.emplace_back(pool.enqueue(zctc::decode<T>, this, logits + ip_pos, ids + ip_pos, labels + op_pos,
                                          timesteps + op_pos, *(seq_len + i),
                                          (hotwords.empty() ? nullptr : &hotwords_fst)));
    }

    for (auto&& result : results)
        if (result.get() != 0)
            throw std::runtime_error("Unexpected error occured during execution");
}

#endif // _ZCTC_DECODER_H
