#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <ThreadPool.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./ext_scorer.hh"
#include "./trie.hh"

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
            char* lm_path, char* lexicon_path)
        : thread_count(thread_count)
        , blank_id(blank_id)
        , cutoff_top_n(cutoff_top_n)
        , vocab_size(vocab.size())
        , nucleus_prob_per_timestep(nucleus_prob_per_timestep)
        , penalty(penalty)
        , beam_width(beam_width)
        , vocab(vocab)
        , ext_scorer(tok_sep, apostrophe_id, lm_alpha, lm_path, lexicon_path)
    {
    }

    template <typename T>
    void batch_decode(py::array_t<T>& batch_log_logits, py::array_t<int>& batch_sorted_ids,
                      py::array_t<int>& batch_labels, py::array_t<int>& batch_timesteps,
                      py::array_t<int>& batch_seq_len, const int batch_size, const int max_seq_len) const;
};

template <typename T>
int
decode(const Decoder* decoder, T* logits, int* ids, int* label, int* timestep, const int seq_len)
{

    T nucleus_max, nucleus_count, prob;
    bool is_repeat, is_blank;
    int iter_val;
    int *curr_id, *curr_l, *curr_t;
    zctc::Node<T>* child;
    std::vector<zctc::Node<T>*> prefixes, tmp;
    zctc::Node<T> root(zctc::ROOT_ID, -1, true, static_cast<T>(zctc::ZERO), static_cast<T>(decoder->penalty), "<s>",
                       nullptr);
    fst::SortedMatcher<fst::StdVectorFst> matcher(decoder->ext_scorer.lexicon, fst::MATCH_INPUT);

    nucleus_max = static_cast<T>(decoder->nucleus_prob_per_timestep);
    decoder->ext_scorer.initialise_start_states(&root);
    prefixes.reserve(decoder->beam_width);
    tmp.reserve(decoder->beam_width);

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

            for (zctc::Node<T>* prefix : prefixes) {

                child = prefix->add_to_child(index, t, prob, decoder->vocab[index], is_blank);
                tmp.push_back(child);
                is_repeat = child->id == prefix->id;

                if (!(is_blank || is_repeat)) {
                    // only run ext scoring for non-duplicate and non-blank tokens
                    decoder->ext_scorer.run_ext_scoring(child, &matcher);

                } else {

                    // only non-blank and repeated token's LM probs will be used
                    if (!is_blank) {
                        child->lm_prob = prefix->lm_prob;
                    }

                    if (!(is_repeat && (child->timestep == prefix->timestep))) {
                        // in case of repeated token but with less confidence, then
                        // the node will not be added to path. Hence the child
                        // will be the prefix(same pointer).
                        child->lm_state = prefix->lm_state;
                        child->lexicon_state = prefix->lexicon_state;
                        child->arc_exist = prefix->arc_exist;
                    }

                    // TODO: try commenting the below line and check...
                    // else {
                    // making the prob of blank node to 0, to avoid pruning of unintended other nodes
                    // child->prob = static_cast<T>(zctc::ZERO);
                    // for blank, the lm_prob will itself be 0
                    // }

                }

                // update total score for the node,
                // considering probs, LM probs, OOV penalty
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

    std::sort(prefixes.begin(), prefixes.end(), Decoder::descending_compare<T>);

    iter_val = 1;
    for (zctc::Node<T>* prefix : prefixes) {

        // NOTE: getting an exception when running the exe `zctc`, might want to have a look
        // prefix var pointing to some meaningless location, for last few elements

        curr_t = timestep + ((seq_len * iter_val) - 1);
        curr_l = label + ((seq_len * iter_val) - 1);

        child = prefix;

        while (child->parent != nullptr) {

            *curr_l = child->id;
            *curr_t = child->timestep;

            // if index becomes less than 0, might throw error
            curr_l--;
            curr_t--;

            child = child->parent;
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
    return x->score > y->score;
}

template <typename T>
void
zctc::Decoder::batch_decode(py::array_t<T>& batch_log_logits, py::array_t<int>& batch_sorted_ids,
                            py::array_t<int>& batch_labels, py::array_t<int>& batch_timesteps,
                            py::array_t<int>& batch_seq_len, const int batch_size, const int max_seq_len) const
{

    py::buffer_info logits_buf = batch_log_logits.request();
    py::buffer_info ids_buf = batch_sorted_ids.request();
    py::buffer_info labels_buf = batch_labels.request();
    py::buffer_info timesteps_buf = batch_timesteps.request();
    py::buffer_info seq_len_buf = batch_seq_len.request();

    if (logits_buf.ndim != 3 || ids_buf.ndim != 3 || labels_buf.ndim != 3 || timesteps_buf.ndim != 3
        || seq_len_buf.ndim != 1)
        throw std::runtime_error("Logits must be three dimensional, like Batch x Seq-len x Vocab, "
                                 "and Sequence Length must be one dimensional, like Batch");

    int* ids = (int*)ids_buf.ptr;
    int* labels = (int*)labels_buf.ptr;
    int* timesteps = (int*)timesteps_buf.ptr;
    int* seq_len = (int*)seq_len_buf.ptr;
    T* logits = (T*)logits_buf.ptr;

    ThreadPool pool(this->thread_count);
    std::vector<std::future<int>> results;

    if (batch_size > 1) {
        for (int i = 1, ip_pos = 0, op_pos = 0; i < batch_size; i++) {
            ip_pos = i * max_seq_len * this->vocab_size;
            op_pos = i * this->beam_width * max_seq_len;

            results.emplace_back(pool.enqueue(zctc::decode<T>, this, logits + ip_pos, ids + ip_pos, labels + op_pos,
                                              timesteps + op_pos, *(seq_len + i)));
        }
    }

    zctc::decode(this, logits, ids, labels, timesteps, *seq_len);

    if (batch_size > 1) {
        for (auto&& result : results)
            if (result.get() != 0)
                throw std::runtime_error("Unexpected error occured during execution");
    }
}

#endif // _ZCTC_DECODER_H
