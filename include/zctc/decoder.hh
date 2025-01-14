#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <ThreadPool.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "./ext_scorer.hh"
#include "./node.hh"
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
            char* lm_path, char* lexicon_path)
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
                      py::array_t<int>& batch_seq_len, py::array_t<int>& batch_seq_pos, const int batch_size,
                      const int max_seq_len, std::vector<std::vector<int>>& hotwords, 
                      std::vector<float>& hotwords_weight) const;
};

template <typename T>
int
decode(const Decoder* decoder, T* logits, int* ids, int* label, int* timestep, const int seq_len, int* seq_pos,
       fst::StdVectorFst* hotwords_fst)
{

    bool is_blank;
    int iter_val, pos_val;
    T nucleus_max, nucleus_count, prob;
    int *curr_id, *curr_l, *curr_t, *curr_p;
    zctc::Node<T>* child;
    std::vector<int> duplicte_ids;
    std::vector<zctc::Node<T>*> prefixes0, prefixes1;
    zctc::Node<T> root(zctc::ROOT_ID, -1, static_cast<T>(zctc::ZERO), "<s>", nullptr);
    fst::SortedMatcher<fst::StdVectorFst> lexicon_matcher(decoder->ext_scorer.lexicon, fst::MATCH_INPUT);
    fst::SortedMatcher<fst::StdVectorFst> hotwords_matcher(hotwords_fst, fst::MATCH_INPUT);

    nucleus_max = static_cast<T>(decoder->nucleus_prob_per_timestep);
    decoder->ext_scorer.initialise_start_states(&root, hotwords_fst);

    // For performance reasons, we initialise and reserve memory
    // for the prefixes
    prefixes0.reserve(decoder->cutoff_top_n * decoder->beam_width);
    prefixes1.reserve(decoder->cutoff_top_n * decoder->beam_width);
    prefixes0.emplace_back(&root);

    for (int t = 0; t < seq_len; t++) {
        // Swap the reader and writer vectors, as per the timestep,
        // to avoid cleaning and copying the elements.
        std::vector<zctc::Node<T>*>& reader = ((t % 2) == 0 ? prefixes0 : prefixes1);
        std::vector<zctc::Node<T>*>& writer = ((t % 2) == 0 ? prefixes1 : prefixes0);

        nucleus_count = 0;
        iter_val = t * decoder->vocab_size;
        curr_id = ids + iter_val;

        for (int i = 0, index = 0; i < decoder->cutoff_top_n; i++, curr_id++) {
            index = *curr_id;
            prob = logits[iter_val + index];

            is_blank = index == decoder->blank_id;
            nucleus_count += prob;
            // prob = std::log(prob);

            if (is_blank) {
                // Just update the blank probs and continue
                // in case of blank.
                for (zctc::Node<T>* r_node : reader) {
                    r_node->_b_prob += prob;
                    r_node->_update_required = true;
                    writer.emplace_back(r_node);
                }

                continue;
            }

            for (zctc::Node<T>* r_node : reader) {
                // Check if the index is repeat or not and 
                // update the node accordingly
                child = r_node->extend_path(index, t, prob, decoder->vocab[index], writer);

                /*
                `nullptr` means the path extension was not done,
                (ie) no new node was created, 
                the probs were accumulated within the current node.
                */
                if (child == nullptr) continue;

                // only newly created nodes are considered for external scoring.
                decoder->ext_scorer.run_ext_scoring(child, &lexicon_matcher, hotwords_fst, &hotwords_matcher);
            }

            if (nucleus_count >= nucleus_max)
                break;
        }

        pos_val = 0;
        for (zctc::Node<T>* w_node : writer) {
            /*
            update total score for the node,
            considering probs, LM probs, OOV penalty
            recently updated token and blank probs
            */

            if (!w_node->_update_required) {
                duplicte_ids.emplace_back(pos_val);
            }
            pos_val++;
            w_node->update_score(decoder->penalty, t);
            /*
            NOTE: Doing the update step here, to avoid
            the current timestep's repeat token prob
            of the node, getting included with a
            different symbol that is getting extended
            in this timestep, like,

                -->        a -  (in this case, the probs will be acc to the curr node itself)
                                (if the prev node has a most recent blank too, then new node)
                                (will also be created and the path will be extended)
                |
            a ---->  (blank) -  (in this case, the probs will be acc to the curr node itself)
                |
                -->        b -  (in this case, a new node is created and the path is extended)

            */

        }

        pos_val = 0;
        for (int pos : duplicte_ids) {
            writer.erase(writer.begin() + pos - pos_val);
            pos_val++;
        }

        duplicte_ids.clear();
        reader.clear();
        if (writer.size() < decoder->beam_width)
            continue;

        std::nth_element(writer.begin(), writer.begin() + decoder->beam_width, writer.end(),
                         Decoder::descending_compare<T>);
        writer.erase(writer.begin() + decoder->beam_width, writer.end());
    }

    std::vector<zctc::Node<T>*>& reader = ((seq_len % 2) == 0 ? prefixes0 : prefixes1);
    std::sort(reader.begin(), reader.end(), Decoder::descending_compare<T>);

    iter_val = 1;
    curr_p = seq_pos;
    for (zctc::Node<T>* r_node : reader) {

        curr_t = timestep + ((seq_len * iter_val) - 1);
        curr_l = label + ((seq_len * iter_val) - 1);
        pos_val = seq_len;

        while (r_node->id != zctc::ROOT_ID) {

            *curr_l = r_node->id;
            *curr_t = r_node->ts;

            curr_l--;
            curr_t--;
            pos_val--;

            r_node = r_node->parent;
        }

        *curr_p = pos_val;
        iter_val++;
        curr_p++;
    }

    return 0;
}

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
bool
zctc::Decoder::descending_compare(zctc::Node<T>* x, zctc::Node<T>* y)
{
    return x->h_score > y->h_score;
}

template <typename T>
void
zctc::Decoder::batch_decode(py::array_t<T>& batch_log_logits, py::array_t<int>& batch_sorted_ids,
                            py::array_t<int>& batch_labels, py::array_t<int>& batch_timesteps,
                            py::array_t<int>& batch_seq_len, py::array_t<int>& batch_seq_pos, const int batch_size,
                            const int max_seq_len, std::vector<std::vector<int>>& hotwords, 
                            std::vector<float>& hotwords_weight) const
{
    ThreadPool pool(std::min(this->thread_count, batch_size));
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
    py::buffer_info seq_pos_buf = batch_seq_pos.request(true);

    if (logits_buf.ndim != 3 || ids_buf.ndim != 3 || labels_buf.ndim != 3 || timesteps_buf.ndim != 3
        || seq_len_buf.ndim != 1 || seq_pos_buf.ndim != 2)
        throw std::runtime_error("Logits must be three dimensional, like Batch x Seq-len x Vocab, "
                                 "and Sequence Length must be one dimensional, like Batch"
                                 "and Sequence Pos mus be two dimensional, like Batch x BeamWidth");

    T* logits = static_cast<T*>(logits_buf.ptr);
    int* ids = static_cast<int*>(ids_buf.ptr);
    int* labels = static_cast<int*>(labels_buf.ptr);
    int* timesteps = static_cast<int*>(timesteps_buf.ptr);
    int* seq_len = static_cast<int*>(seq_len_buf.ptr);
    int* seq_pos = static_cast<int*>(seq_pos_buf.ptr);

    for (int i = 0, ip_pos = 0, op_pos = 0, s_p = 0; i < batch_size; i++) {
        ip_pos = i * max_seq_len * this->vocab_size;
        op_pos = i * this->beam_width * max_seq_len;
        s_p = i * this->beam_width;

        results.emplace_back(pool.enqueue(zctc::decode<T>, this, logits + ip_pos, ids + ip_pos, labels + op_pos,
                                          timesteps + op_pos, *(seq_len + i), seq_pos + s_p,
                                          (hotwords.empty() ? nullptr : &hotwords_fst)));
    }

    for (auto&& result : results)
        if (result.get() != 0)
            throw std::runtime_error("Unexpected error occured during execution");
}

#endif // _ZCTC_DECODER_H
