#ifndef _ZCTC_TRIE_H
#define _ZCTC_TRIE_H

#include <algorithm>
#include <string>
#include <vector>

#include "./constants.hh"

namespace zctc {

template<typename T>
class Node {
public:

    bool still_in_prefixes;
    int id, timestep;
    T prob, parent_scr, lm_prob, score;
    Node<T>* parent;
    std::vector<std::string> n_tokens;
    std::vector<Node<T>*> childs;

    Node(int id, int timestep, T prob, Node<T>* parent, int context_size, bool is_repeat_node)
        : still_in_prefixes(true),
          id(id),
          timestep(timestep),
          prob(prob),
          parent_scr(static_cast<T>(zctc::ZERO)),
          lm_prob(static_cast<T>(zctc::ZERO)),
          score(static_cast<T>(zctc::ZERO)),
          parent(parent)
    {
        if (this->parent == nullptr) {
            this->n_tokens.resize(context_size, "<s>");
            return;
        }

        this->parent_scr = parent->score;
        if (is_repeat_node) {
            return;
        } 

        this->n_tokens.reserve(context_size);
        std::copy_n(this->parent->n_tokens.begin() + 1, context_size, this->n_tokens.begin());
        // this->n_tokens[context_size] = this->id_to_tok;

        this->update_score();
    }

    ~Node() {
        for (Node<T>* child : *this)
            delete child;
    }

    inline void update_score() noexcept;
    inline void start_token_check() noexcept;

    Node<T>* add_to_child(int id, int timestep, T prob, int context_size);

    // element-wise iterator for this class,
    typename std::vector<Node<T>*>::iterator begin() noexcept { return this->childs.begin(); }
    typename std::vector<Node<T>*>::const_iterator cbegin() const noexcept { return this->childs.cbegin(); }
    typename std::vector<Node<T>*>::iterator end() noexcept { return this->childs.end(); }
    typename std::vector<Node<T>*>::const_iterator cend() const noexcept { return this->childs.cend(); }
};

} // namespace zctc


/* ---------------------------------------------------------------------------- */


template <typename T>
void zctc::Node<T>::update_score() noexcept {
    this->score = this->parent_scr + this->prob + this->lm_prob;
}

template <typename T>
void zctc::Node<T>::start_token_check() noexcept {
    // ...;
}

template <typename T>
zctc::Node<T>* zctc::Node<T>::add_to_child(int id, int timestep, T prob, int context_size) {

    Node<T>* child;

    if (id == this->id) {
        
        child = new Node<T>(id, timestep, prob + this->prob, this->parent, context_size, true);
        child->lm_prob = this->lm_prob;

        child->update_score();

        child->n_tokens = this->n_tokens;
        this->parent->childs.push_back(child);

    } else {

        child = new Node<T>(id, timestep, prob, this, context_size, false);
        this->childs.push_back(child);

    }

    return child;
}


#endif // _ZCTC_TRIE_H