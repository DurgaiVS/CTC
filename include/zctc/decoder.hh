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
        py::array_t<T>& log_logits,
        py::array_t<int>& sorted_ids,
        py::array_t<int>& labels,
        py::array_t<int>& timesteps,
        int seq_len
    ) const;

    // template <typename T>
    // void batch_decode(
    //     py::array_t<T>& batch_log_logits,
    //     py::array_t<int>& batch_sorted_ids,
    //     py::array_t<int>& batch_labels,
    //     py::array_t<int>& batch_timesteps,
    //     int batch_size,
    //     int seq_len
    // ) const {
    //     auto logits = batch_log_logits.mutable_unchecked();
    //     auto sorted_ids = batch_sorted_ids.mutable_unchecked<3>();
    //     auto labels = batch_labels.mutable_unchecked<3>();
    //     auto timesteps = batch_timesteps.mutable_unchecked<3>();

    //     for (int i = 0; i < batch_log_logits.shape(0); i++) {
    //         this->decode(logits(i), sorted_ids(i), labels(i), timesteps(i), seq_len);
    //     }

    // }

};

/* ---------------------------------------------------------------------------- */

template <typename T>
bool Decoder::descending_compare(Node<T>* x, Node<T>* y) {
    if (x->score == y->score) {
        if (x->id == y->id) {
            return false;
        } else {
            return (x->id < y->id);
        }
    } else {
        return x->score > y->score;
    }
}

template <typename T>
void Decoder::decode(
        py::array_t<T>& log_logits,
        py::array_t<int>& sorted_ids,
        py::array_t<int>& labels,
        py::array_t<int>& timesteps,
        int seq_len
) const {

    py::buffer_info logits_buf = log_logits.request();
    py::buffer_info ids_buf = sorted_ids.request();
    py::buffer_info labels_buf = labels.request();
    py::buffer_info timesteps_buf = timesteps.request();

    if (logits_buf.ndim != 2 || ids_buf.ndim != 2 || labels_buf.ndim != 2 || timesteps_buf.ndim != 2)
        throw std::runtime_error("number of dimensions must be two");

    T nucleus_max = static_cast<T>(this->nucleus_prob_per_timestep);
    std::vector<Node<T>*> prefixes, tmp;
    prefixes.reserve(this->beam_width);
    tmp.reserve(this->beam_width);

    Node<T> root(_ZCTC_ROOT_ID, -1, static_cast<T>(_ZCTC_ZERO), static_cast<T>(_ZCTC_ZERO), nullptr);
    prefixes.push_back(&root);

    Node<T>* child;
    int *curr_id, *curr_l, *curr_t;
    int *ids = (int*)ids_buf.ptr, *label = (int*)labels_buf.ptr, *timestep = (int*)timesteps_buf.ptr;
    T* logits = (T*)logits_buf.ptr;
    int t_val;

    for (int t = 0; t < seq_len; t++) {
        T nucleus_count = 0;
        t_val = t * this->vocab_size;
        curr_id = ids + t_val;

        for (int i = 0; i < this->cutoff_top_n; i++) {
            int index = *curr_id;
            T prob = logits[t_val + index];

            nucleus_count += prob;

            for (Node<T>* prefix : prefixes) {

                if (index == this->blank_id) {
                    prefix->b_prob += prob;
                    prefix->update_score();
                } else if (index == prefix->id) {
                    if (prefix->nb_prob < prob) {
                        prefix->nb_prob = prob;
                        prefix->timestep = t;
                    }
                    prefix->update_score();
                } else {
                    child = prefix->add_to_child(index, t, prob);
                    if (child != nullptr) 
                        tmp.push_back(child);
                }
            }

            curr_id++;
            if (nucleus_count >= nucleus_max) break;
        }

        std::copy(tmp.begin(), tmp.end(), std::back_inserter(prefixes));
        tmp.clear();
        tmp.reserve(this->beam_width);

        if (prefixes.size() < this->beam_width)
            continue;

        std::nth_element(prefixes.begin(), prefixes.begin() + this->beam_width, prefixes.end(), Decoder::descending_compare<T>);
        std::for_each(prefixes.begin() + this->beam_width, prefixes.end(), [&](Node<T>* node) {
            node->stash_node();
        });

        prefixes.erase(prefixes.begin() + this->beam_width, prefixes.end());
    }

    std::sort(prefixes.begin(), prefixes.end(), Decoder::descending_compare<T>);

        /*
        m.def("increment_3d", [](py::array_t<T> x) {
        auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
        for (int i = 0; i < r.shape(0); i++)
            for (int j = 0; j < r.shape(1); j++)
                for (int k = 0; k < r.shape(2); k++)
                    r(i, j, k) += 1.0;
        }, py::arg().noconvert());
        */

    // static_assert(this->beam_width == labels_buf.shape[0], "Labels array should be of shape (beam_width X seq_len)");
    // static_assert(this->beam_width == timesteps_buf.shape[0], "Timesteps array should be of shape (beam_width X seq_len)");

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