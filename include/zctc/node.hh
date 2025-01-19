#ifndef _ZCTC_TRIE_H
#define _ZCTC_TRIE_H

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "lm/state.hh"

#include "./constants.hh"

namespace zctc {

template <typename T>
class Node {
public:
    bool is_lex_path, is_start_of_word, is_hotpath, _update_required;
    int hotword_length, ts, b_ts, tk_ts;
    T _tk_prob, _b_prob, _intrm_score;
    T max_prob, _max_prob, p_score, lm_prob, score, h_score, hotword_weight;
    const int id;
    const std::string token;
    Node<T>* parent;
    lm::ngram::State lm_state;
    fst::StdVectorFst::StateId lexicon_state, hotword_state;
    std::vector<Node<T>*> childs;

    Node(int id, int ts, T prob, const std::string token, Node<T>* parent)
        : is_lex_path(true)
        , is_start_of_word(false)
        , is_hotpath(false)
        , _update_required(true)
        , hotword_length(0)
        , ts(ts)
        , b_ts(-1)
        , tk_ts(ts)
        , _tk_prob(prob)
        , _b_prob(zctc::ZERO)
        , _intrm_score(zctc::ZERO)
        , max_prob(prob)
        , _max_prob(prob)
        , p_score(zctc::ZERO)
        , lm_prob(zctc::ZERO)
        , score(zctc::ZERO)
        , h_score(zctc::ZERO)
        , hotword_weight(zctc::ZERO)
        , id(id)
        , token(token)
        , parent(parent)
    {
        if (this->parent == nullptr)
            return;

        this->p_score = parent->score;
    }

    // Copy Constructor
    Node(const Node& other)
        : is_lex_path(other.is_lex_path)
        , is_start_of_word(other.is_start_of_word)
        , is_hotpath(other.is_hotpath)
        , _update_required(other._update_required)
        , hotword_length(other.hotword_length)
        , ts(other.ts)
        , b_ts(other.b_ts)
        , tk_ts(other.tk_ts)
        , _tk_prob(other._tk_prob)
        , _b_prob(other._b_prob)
        , _intrm_score(other._intrm_score)
        , max_prob(other.max_prob)
        , _max_prob(other._max_prob)
        , p_score(other.p_score)
        , lm_prob(other.lm_prob)
        , score(other.score)
        , h_score(other.h_score)
        , hotword_weight(other.hotword_weight)
        , id(other.id)
        , token(other.token)
        , parent(other.parent)
    { }

    ~Node()
    {
        for (Node<T>* child : *this)
            delete child;
    }

    inline void acc_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
    inline void acc_tk_and_parent_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
    inline void update_score(float penalty, int curr_ts) noexcept;

    Node<T>* extend_path(int id, int ts, T prob, const std::string token, std::vector<Node<T>*>& writer);
    inline Node<T>* acc_repeat_token_prob(int ts, T prob, std::vector<Node<T>*>& writer) noexcept;

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
zctc::Node<T>::update_score(float penalty, int curr_ts) noexcept
{
    /*
        In case, if a node encounters a blank and a repeat token
        at the same timestep, this could cause the node to have
        two calls to this function, which could lead to the
        node's score to be updated twice, which is not desirable.

        To avoid a check when inserting the nodes into the writer,
        we are going with this approach.
    */
    if (!this->_update_required) return;

    if (this->_max_prob != this->max_prob) {
        this->max_prob = this->_max_prob;
        this->ts = curr_ts;
    }

    /*
        Here, the `_tk_prob` and `_b_prob` and `lm_prob` will be in
        linear scale, `p_score` and `score` will be in log scale,
    */
    this->_intrm_score += std::log(this->_tk_prob + this->_b_prob);
    if (this->lm_prob != zctc::ZERO) this->score = std::log(std::exp(this->p_score + this->_intrm_score) + this->lm_prob);
    else this->score = this->p_score + this->_intrm_score;

    if (!this->is_lex_path) this->score += penalty;

    if (this->is_hotpath) {
        this->h_score = this->score + (this->hotword_length * this->hotword_weight);
    } else {
        this->h_score = this->score;
    }

    this->_tk_prob = zctc::ZERO;
    this->_b_prob = zctc::ZERO;
    this->_update_required = false;
}

template <typename T>
void
zctc::Node<T>::acc_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
    zctc::Node<T>* node = this;
    if (prob > this->_max_prob) {
        node = new zctc::Node<T>(*this);
        node->_max_prob = prob;
    }
    node->_tk_prob += prob;
    node->_update_required = true;

    writer.emplace_back(node);
}

template <typename T>
void
zctc::Node<T>::acc_tk_and_parent_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
    zctc::Node<T>* node = this;
    if (prob > this->_max_prob) {
        node = new zctc::Node<T>(*this);
        node->_max_prob = prob;
    }

    node->_update_required = true;
    writer.emplace_back(node);

    if (node->parent->score == node->p_score) {
        node->_tk_prob += prob;
        return;
    }

    T diff_prob = node->p_score - node->parent->score;
    node->_tk_prob += std::exp(diff_prob + std::log(prob));
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::acc_repeat_token_prob(int ts, T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
    /*
        In case, if the token is the most recent than the blank, or,
        if the blank and token are at the same timestep,
        within this node,
        we can accumulate the probs with the current node.

        The assumption here is,
        If the token is the most recent than the blank, then we will
        treat the token as monotonic, and we won't extend the path,
        rather we will accumulate the probs with the current node.
    */
    if (this->tk_ts >= this->b_ts) this->acc_prob(prob, writer);

    /*
        In case, if the blank is more recent than the token, or,
        if the blank and token are at the same timestep,
        within this node,
        we can create a new child, assuming that the current token
        (the one passed in the argument) is preceded by a blank.

        The assumption here is,
        If the blank is the most recent than the token, then we will
        treat the token as non-monotonic, and we will extend the path,
        rather than accumulating the probs with the current node, due to
        the fact that the token is preceded by a blank.
    */
    if (this->b_ts >= this->tk_ts) {
        /*
            In case, if the child is already available, we can just update
            the probs and return the child.
            Not sure if this case is possible or not, but just wanted to
            ensure that we are not creating duplicate child nodes.
        */
        for (zctc::Node<T>* r_node : *this) {
            if (r_node->id != id) continue;

            r_node->acc_prob(prob, writer);
            return nullptr;
        }

        // If the child is not available, then we can create a new child.
        zctc::Node<T>* child = new zctc::Node<T>(id, ts, prob, token, this);

        this->childs.emplace_back(child);
        writer.emplace_back(child);
        return child;
    }

    return nullptr;
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::extend_path(int id, int ts, T prob, const std::string token, std::vector<zctc::Node<T>*>& writer)
{
    /*
        TODO: In case of repeat, but with most confident prob,
        then, try to create a new node, coz updating in this node
        will affect the previously added child's timesteps.
    */

    if (id == this->id) 
        return this->acc_repeat_token_prob(ts, prob, writer);

    for (Node<T>* r_node : *this) {
        if (r_node->id != id) continue;

        /*
            If the current node has a child with the provided id,
            then we can accumulate the probs within the child node.

            Since, we're accumulating the repeated and blank probs
            within the node itself, there could be a possibility that
            the parent's probs have changed from the timestep this
            child was created, so we need to update the parent probs
            value in the child node too...
        */
        r_node->acc_tk_and_parent_prob(prob, writer);
        return nullptr;
    }

    // If the current node has no child with the provided id,
    // then we can create a new child.
    zctc::Node<T>* child = new zctc::Node<T>(id, ts, prob, token, this);

    this->childs.emplace_back(child);
    writer.emplace_back(child);
    return child;
}

#endif // _ZCTC_TRIE_H
