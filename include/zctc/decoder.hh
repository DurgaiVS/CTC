#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include <algorithm>
#include "./trie.hh"

class Decoder {
public:
    template <typename T>
    static bool descending_compare(Node<T>* x, Node<T>* y);

    const int blank_id, cutoff_top_n, thread_count;
    const float nucleus_prob_per_timestep;
    const std::size_t beam_width;

    Decoder(int blank_id, std::size_t beam_width, int cutoff_top_n, int thread_count, float nucleus_prob_per_timestep)
        : blank_id(blank_id),
          cutoff_top_n(cutoff_top_n),
          thread_count(thread_count),
          nucleus_prob_per_timestep(nucleus_prob_per_timestep),
          beam_width(beam_width)
    { }

    template <typename T>
    void decode(
        std::vector<std::vector<T>>& log_logits,
        std::vector<std::vector<int>>& sorted_ids,
        std::vector<std::vector<int>>& labels,
        std::vector<std::vector<int>>& timesteps,
        int seq_len
    ) const;

    template <typename T>
    void batch_decode(
        std::vector<std::vector<std::vector<T>>>& batch_log_logits,
        std::vector<std::vector<std::vector<int>>>& batch_sorted_ids,
        std::vector<std::vector<std::vector<int>>>& batch_labels,
        std::vector<std::vector<std::vector<int>>>& batch_timesteps,
        int batch_size,
        int seq_len
    ) const {

        for (int i = 0; i < batch_size; i++) {
            this->decode(batch_log_logits[i], batch_sorted_ids[i], batch_labels[i], batch_timesteps[i], seq_len);
        }

    }

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
        std::vector<std::vector<T>>& log_logits,
        std::vector<std::vector<int>>& sorted_ids,
        std::vector<std::vector<int>>& labels,
        std::vector<std::vector<int>>& timesteps,
        int seq_len
    ) const {

    T nucleus_max = static_cast<T>(this->nucleus_prob_per_timestep);
    std::vector<Node<T>*> prefixes, tmp;
    prefixes.reserve(this->beam_width);
    tmp.reserve(this->beam_width);

    Node<T> root(_ZCTC_ROOT_ID, -1, static_cast<T>(_ZCTC_ZERO), static_cast<T>(_ZCTC_ZERO), nullptr);
    prefixes.push_back(&root);

    std::vector<T>* log_logit = log_logits.data();
    std::vector<int>* sorted_id = sorted_ids.data();
    Node<T>* child;
    for (int timestep = 0; timestep < seq_len; timestep++, log_logit++, sorted_id++) {
        T nucleus_count = 0;

        for (int i = 0; i < this->cutoff_top_n; i++) {
            int index = (*sorted_id)[i];
            T prob = (*log_logit)[index];

            nucleus_count += prob;

            for (Node<T>* prefix : prefixes) {

                if (index == this->blank_id) {
                    prefix->b_prob += prob;
                    prefix->update_score();
                } else if (index == prefix->id) {
                    if (prefix->nb_prob < prob) {
                        prefix->nb_prob = prob;
                        prefix->timestep = timestep;
                    }
                    prefix->update_score();
                } else {
                    child = prefix->add_to_child(index, timestep, prob);
                    if (child != nullptr) 
                        tmp.push_back(child);
                }
            }
            // --- OP ---

            // save the prefixes to a vector, with max size of beam_width
            // keep sorting in descending and prune nodes which lies out
            // of the range.

            // Keep adding nodes to this prefix vector, and once a timestep
            // is over, prune outliers.

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

    int i = 0, j;
    for (Node<T>* prefix : prefixes) {
        j = labels[i].size() - 1;

        while (prefix->parent != nullptr) {
            labels[i][j] = prefix->id;
            timesteps[i][j] = prefix->timestep;
            j--;
            // if i becomes less than 0, might throw error

            prefix = prefix->parent;
        }

        i++;
    }

}

#endif // _ZCTC_DECODER_H