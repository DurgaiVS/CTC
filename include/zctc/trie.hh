#ifndef _ZCTC_TRIE_H
#define _ZCTC_TRIE_H

#include <algorithm>
#include <vector>

#include "./constants.hh"

template<typename T>
class Node {
public:

    bool still_in_prefixes;
    int id, timestep;
    T b_prob, nb_prob, parent_scr, lm_prob, score;
    Node<T>* parent;
    std::vector<Node<T>*> childs;

    Node(int id, int timestep, T b_prob, T nb_prob, Node<T>* parent)
        : still_in_prefixes(true),
          id(id),
          timestep(timestep),
          b_prob(b_prob),
          nb_prob(nb_prob),
          parent_scr(static_cast<T>(_ZCTC_ZERO)),
          lm_prob(static_cast<T>(_ZCTC_ZERO)),
          score(static_cast<T>(_ZCTC_ZERO)),
          parent(parent)
    {
        if (this->parent != nullptr)
            this->parent_scr = parent->score;
    }

    ~Node() {
        for (Node<T>* child : *this)
            delete child;
    }

    void stash_node();
    inline void update_score() noexcept;

    void get_full_path(std::vector<int>& labels, std::vector<int>& timesteps);
    void get_prefixes(int count, int* suffixes, int pad_tok_id);
    Node<T>* add_to_child(int id, int timestep, T prob);

    // element-wise iterator for this class,
    typename std::vector<Node<T>*>::iterator begin() noexcept { return this->childs.begin(); }
    typename std::vector<Node<T>*>::const_iterator cbegin() const noexcept { return this->childs.cbegin(); }
    typename std::vector<Node<T>*>::iterator end() noexcept { return this->childs.end(); }
    typename std::vector<Node<T>*>::const_iterator cend() const noexcept { return this->childs.cend(); }
};

/* ---------------------------------------------------------------------------- */

template <typename T>
Node<T>* Node<T>::add_to_child(int id, int timestep, T prob) {
    for (Node<T>* child : *this) {
        if (id == child->id) {
            if (child->nb_prob < prob) {
                child->nb_prob = prob;
                child->timestep = timestep;

                child->update_score();

            }
            return nullptr;
        }
    }
    Node<T>* child = new Node<T>(id, timestep, static_cast<T>(_ZCTC_ZERO), prob, this);
    this->childs.push_back(child);

    return child;
}

template <typename T>
void Node<T>::stash_node() {
    this->still_in_prefixes = false;

    if (this->childs.size() != 0) return;

    this->parent->childs.erase(
        std::remove_if(this->parent->childs.begin(), this->parent->childs.end(), [&](Node<T>* child) {
            return this->id == child->id;
        })
    );

    // if (this->parent->childs.size() == 0) // !this->parent->still_in_prefixes && (check once if this is needed)
    //     this->parent->stash_node();

    delete this;
}

template <typename T>
void Node<T>::update_score() noexcept {
    this->score = this->parent_scr + this->b_prob + this->nb_prob + this->lm_prob;
}

template <typename T>
void Node<T>::get_full_path(std::vector<int>& labels, std::vector<int>& timesteps) {
    labels.push_back(this->id);
    timesteps.push_back(this->timestep);

    if (this->parent == nullptr) {
        std::reverse(labels.begin(), labels.end());
        std::reverse(timesteps.begin(), timesteps.end());
        return;
    }

    return this->parent->get_full_path(labels, timesteps);
}

template <typename T>
void Node<T>::get_prefixes(int count, int* suffixes, int pad_tok_id) {
    Node<T>* temp = this;
    for (; count > 0; count--) {
        if (temp->id == _ZCTC_ROOT_ID) {
            suffixes[count] = pad_tok_id;
        } else {
            suffixes[count] = temp->id;
            temp = temp->parent;
        }
    }
}

#endif // _ZCTC_TRIE_H