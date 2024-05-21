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
    bool arc_exist, is_start_of_word;
    const int id, timestep;
    T prob, parent_scr, lm_prob, score, penalty, max_prob;
    const std::string& token;
    Node<T>* parent;
    lm::ngram::State lm_state;
    fst::StdVectorFst::StateId lexicon_state;
    std::vector<Node<T>*> childs;

    Node(int id, int timestep, T prob, T penalty, const std::string& token, Node<T>* parent)
        : arc_exist(false)
        , id(id)
        , timestep(timestep)
        , prob(prob)
        , penalty(penalty)
        , max_prob(prob)
        , token(token)
        , parent_scr(static_cast<T>(zctc::ZERO))
        , lm_prob(static_cast<T>(zctc::ZERO))
        , score(prob)
        , parent(parent)
    {
        if (this->parent == nullptr)
            return;

        this->parent_scr = parent->score;

        this->parent->childs.push_back(this);
    }

    Node(int id, int timestep, T prob, T penalty, T max_prob, const std::string& token, Node<T>* parent)
        : arc_exist(false)
        , id(id)
        , timestep(timestep)
        , prob(prob)
        , penalty(penalty)
        , max_prob(max_prob)
        , token(token)
        , parent_scr(static_cast<T>(zctc::ZERO))
        , lm_prob(static_cast<T>(zctc::ZERO))
        , score(prob)
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

    inline void update_score(bool is_blank) noexcept;

    Node<T>* add_to_child(int id, int timestep, T prob, const std::string& token, bool* is_repeat);

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
zctc::Node<T>::update_score(bool is_blank) noexcept
{
    this->score = this->parent_scr + this->prob + this->lm_prob;

    if (!(this->arc_exist || is_blank))
        this->score += this->penalty;
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::add_to_child(int id, int timestep, T prob, const std::string& token, bool* is_repeat)
{

    Node<T>* child;

    if (id == this->id) {

        T max_prob = prob;

        if (prob < this->max_prob) {
            timestep = this->timestep;
            max_prob = this->max_prob;
        }

        *is_repeat = true;
        child = new Node<T>(id, timestep, prob + this->prob, this->penalty, max_prob, token, this->parent);

    } else {

        *is_repeat = false;
        child = new Node<T>(id, timestep, prob, this->penalty, token, this);
    }

    return child;
}

#endif // _ZCTC_TRIE_H
