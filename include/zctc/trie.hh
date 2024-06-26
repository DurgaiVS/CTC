#ifndef _ZCTC_TRIE_H
#define _ZCTC_TRIE_H

#include <algorithm>
#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "lm/state.hh"

#include "./constants.hh"

namespace zctc {

template <typename T>
class Node {
public:
    bool arc_exist, is_start_of_word, is_blank;
    const int id, timestep;
    T prob, parent_scr, lm_prob, score, penalty;
    const std::string token;
    Node<T>* parent;
    lm::ngram::State lm_state;
    fst::StdVectorFst::StateId lexicon_state;
    std::vector<Node<T>*> childs;

    Node(int id, int timestep, bool is_blank, T prob, T penalty, const std::string token, Node<T>* parent)
        : arc_exist(false)
        , is_blank(is_blank)
        , id(id)
        , timestep(timestep)
        , prob(prob)
        , penalty(penalty)
        , token(token)
        , parent_scr(static_cast<T>(zctc::ZERO))
        , lm_prob(static_cast<T>(zctc::ZERO))
        , score(static_cast<T>(zctc::ZERO))
        , parent(parent)
    {
        if (this->parent == nullptr)
            return;

        this->parent_scr = parent->score;

        this->parent->childs.push_back(this);
    }

    ~Node()
    {
        for (Node<T>* child : *this)
            delete child;
    }

    inline void update_score() noexcept;

    Node<T>* add_to_child(int id, int timestep, T prob, const std::string token, bool is_blank);

    // element-wise iterator for this class,
    typename std::vector<Node<T>*>::iterator begin() noexcept { return this->childs.begin(); }
    typename std::vector<Node<T>*>::iterator end() noexcept { return this->childs.end(); }
    typename std::vector<Node<T>*>::const_iterator cbegin() const noexcept { return this->childs.cbegin(); }
    typename std::vector<Node<T>*>::const_iterator cend() const noexcept { return this->childs.cend(); }
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
void
zctc::Node<T>::update_score() noexcept
{
    this->score = this->parent_scr + this->prob + this->lm_prob;

    if (!(this->arc_exist || this->is_blank))
        this->score += this->penalty;
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::add_to_child(int id, int timestep, T prob, const std::string token, bool is_blank)
{

    Node<T>* child;

    if (id == this->id) {

        if (prob > this->prob) {
            child = new Node<T>(id, timestep, is_blank, prob, this->penalty, token, this->parent);
        } else {
            child = this;
        }

    } else {
        child = new Node<T>(id, timestep, is_blank, prob, this->penalty, token, this);
    }

    return child;
}

#endif // _ZCTC_TRIE_H
