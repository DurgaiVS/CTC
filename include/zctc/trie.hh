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
    T prob, parent_scr, lm_prob, score;
    Node<T>* parent;
    std::vector<Node<T>*> childs;

    Node(int id, int timestep, T prob, Node<T>* parent)
        : still_in_prefixes(true),
          id(id),
          timestep(timestep),
          prob(prob),
          parent_scr(static_cast<T>(_ZCTC_ZERO)),
          lm_prob(static_cast<T>(_ZCTC_ZERO)),
          score(static_cast<T>(_ZCTC_ZERO)),
          parent(parent)
    {
        if (this->parent != nullptr)
            this->parent_scr = parent->score;

        this->update_score();
    }

    ~Node() {
        for (Node<T>* child : *this)
            delete child;
    }

    inline void update_score() noexcept;

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

    Node<T>* child = new Node<T>(id, timestep, prob, this);

    if (id == this->id) {

        child->parent = this->parent;
        child->lm_prob = this->lm_prob;
        child->update_score();
        this->parent->childs.push_back(child);

    } else {

        this->childs.push_back(child);

    }

    return child;
}

template <typename T>
void Node<T>::update_score() noexcept {
    this->score = this->parent_scr + this->prob + this->lm_prob;
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