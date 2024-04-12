#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "./trie.hh"

namespace py = pybind11;

class Decoder {
public:
    template <typename T>
    static bool descending_compare(Node<T>* x, Node<T>* y);

    const int thread_count, blank_id, cutoff_top_n, vocab_size;
    const float nucleus_prob_per_timestep;
    const std::size_t beam_width;

    Decoder(int thread_count, int blank_id, int cutoff_top_n, int vocab_size, float nucleus_prob_per_timestep, std::size_t beam_width)
        : thread_count(thread_count),
          blank_id(blank_id),
          cutoff_top_n(cutoff_top_n),
          vocab_size(vocab_size),
          nucleus_prob_per_timestep(nucleus_prob_per_timestep),
          beam_width(beam_width)
    { }

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

/* ---------------------------------------------------------------------------- */

template <typename T>
bool Decoder::descending_compare(Node<T>* x, Node<T>* y) {
    return x->score > y->score;
}

template <typename T>
void Decoder::decode(
    T* logits,
    int* ids,
    int* label,
    int* timestep,
    const int seq_len
) const {

    T nucleus_max = static_cast<T>(this->nucleus_prob_per_timestep);
    std::vector<Node<T>*> prefixes, tmp;
    prefixes.reserve(this->beam_width);
    tmp.reserve(this->beam_width);

    Node<T> root(_ZCTC_ROOT_ID, -1, static_cast<T>(_ZCTC_ZERO), nullptr);
    prefixes.push_back(&root);

    Node<T>* child;
    int *curr_id, *curr_l, *curr_t;
    int t_val;

    for (int t = 0; t < seq_len; t++) {
        T nucleus_count = 0;
        t_val = t * this->vocab_size;
        curr_id = ids + t_val;

        for (int i = 0; i < this->cutoff_top_n; i++, curr_id++) {
            int index = *curr_id;
            T prob = logits[t_val + index];

            nucleus_count += prob;

            for (Node<T>* prefix : prefixes) {

                if (index == prefix->id) {

                    if (prefix->prob < prob) {
                        prefix->prob = prob;
                        prefix->timestep = t;

                        prefix->update_score();
                    }
                    tmp.push_back(prefix);
                    continue;

                } else {

                    child = prefix->add_to_child(index, t, prob); 
                    tmp.push_back(child);
                    continue;

                }
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
        tmp.reserve(this->beam_width);

    }

    std::sort(prefixes.begin(), prefixes.end(), Decoder::descending_compare<T>);

    t_val = 1;
    for (Node<T>* prefix : prefixes) {
        curr_t = timestep + ((seq_len * t_val) - 1);
        curr_l = label + ((seq_len * t_val) - 1);

        while (prefix->parent != nullptr) {
            *curr_l = prefix->id;
            *curr_t = prefix->timestep;

            // if index becomes less than 0, might throw error
            curr_l--;
            curr_t--;

            prefix = prefix->parent;
        }

        t_val++;
    }

}

#endif // _ZCTC_DECODER_H