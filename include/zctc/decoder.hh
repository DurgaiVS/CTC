#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "./trie.hh"

namespace py = pybind11;

namespace zctc {

class Decoder {
public:
    template <typename T>
    static bool descending_compare(zctc::Node<T>* x, zctc::Node<T>* y);

    const int thread_count, blank_id, cutoff_top_n, vocab_size;
    const float nucleus_prob_per_timestep;
    const std::size_t beam_width;

    int n_gram, n_context;

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

} // namespace zctc

/* ---------------------------------------------------------------------------- */


template <typename T>
bool zctc::Decoder::descending_compare(zctc::Node<T>* x, zctc::Node<T>* y) {
    return x->score > y->score;
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
    std::vector<zctc::Node<T>*> prefixes, tmp;
    prefixes.reserve(this->beam_width);
    tmp.reserve(this->beam_width);

    zctc::Node<T> root(zctc::ROOT_ID, -1, static_cast<T>(zctc::ZERO), nullptr, this->n_context, false);
    prefixes.push_back(&root);

    zctc::Node<T>* child;
    int *curr_id, *curr_l, *curr_t;
    int iter_val;

    for (int t = 0; t < seq_len; t++) {
        T nucleus_count = 0;
        iter_val = t * this->vocab_size;
        curr_id = ids + iter_val;

        for (int i = 0; i < this->cutoff_top_n; i++, curr_id++) {
            int index = *curr_id;
            T prob = logits[iter_val + index];

            nucleus_count += prob;

            for (zctc::Node<T>* prefix : prefixes) {

                child = prefix->add_to_child(index, t, prob, this->n_context); 
                tmp.push_back(child);

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