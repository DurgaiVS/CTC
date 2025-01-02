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
    bool is_lex_path, is_start_of_word, is_hotpath;
    int hotword_length, ts, b_ts, tk_ts;
    T tk_prob, b_prob, _tk_prob, _b_prob;
    T max_prob, parent_scr, lm_prob, score, score_w_h, hotword_weight;
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
        , hotword_length(0)
        , ts(ts)
        , b_ts(0)
        , tk_ts(ts)
        , tk_prob(prob)
        , b_prob(zctc::ZERO)
        , _tk_prob(zctc::ZERO)
        , _b_prob(zctc::ZERO)
        , max_prob(prob)
        , parent_scr(zctc::ZERO)
        , lm_prob(zctc::ZERO)
        , score(zctc::ZERO)
        , score_w_h(zctc::ZERO)
        , hotword_weight(zctc::ZERO)
        , id(id)
        , token(token)
        , parent(parent)
    {
        if (this->parent == nullptr)
            return;

        this->parent_scr = parent->score;
    }

    ~Node()
    {
        for (Node<T>* child : *this)
            delete child;
    }

    // TODO: Try to get the penalty as an argument to this function.
    // BTW, we can also get rid of the penalty attribute.
    inline void update_score(float penalty) noexcept;

    Node<T>* extend_path(int id, int ts, T prob, const std::string token, std::vector<Node<T>*>& writer);
    inline void acc_repeat_token_prob(int ts, T prob) noexcept;

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
zctc::Node<T>::update_score(float penalty) noexcept
{
    if ((this->tk_ts != this->ts) && (this->_tk_prob > this->max_prob)) {
        this->max_prob = this->_tk_prob;
        this->ts = this->tk_ts;
    }

    this->tk_prob += this->_tk_prob;
    this->b_prob += this->_b_prob;

    this->score = this->parent_scr + this->tk_prob + this->b_prob + this->lm_prob;

    if (!this->is_lex_path) this->score += penalty;

    if (this->is_hotpath) {
        this->score_w_h = this->score + (this->hotword_length * this->hotword_weight);
    } else {
        this->score_w_h = this->score;
    }

    this->_tk_prob = zctc::ZERO;
    this->_b_prob = zctc::ZERO;
}

template <typename T>
void 
zctc::Node<T>::acc_repeat_token_prob(int ts, T prob) noexcept
{
    this->_tk_prob = prob;
    this->tk_ts = ts;
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::extend_path(int id, int ts, T prob, const std::string token, std::vector<Node<T>*>& writer)
{

    Node<T>* child;

    if (id == this->id) {
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
        if (this->tk_ts >= this->b_ts) {
            this->acc_repeat_token_prob(ts, prob);
            writer.push_back(this);
        }

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
        if ((this->b_prob != zctc::ZERO) && (this->b_ts >= this->tk_ts)) {
            /*
            In case, if the child is already available, we can just update
            the probs and return the child.
            Not sure if this case is possible or not, but just wanted to
            ensure that we are not creating duplicate child nodes.
            */
            for (Node<T>* r_node : *this) {
                if (r_node->id != id) continue;

                r_node->tk_prob += prob;
                if (ts > r_node->tk_ts) r_node->tk_ts = ts;

                writer.push_back(r_node);
                return nullptr;
            }

            // If the child is not available, then we can create a new child.
            child = new Node<T>(id, ts, prob, token, this);

            this->childs.push_back(child);
            writer.push_back(child);
            return child;
        }

        return nullptr;
    }

    for (Node<T>* r_node : *this) {
        if (r_node->id != id) continue;

        // If the current node has a child with the provided id,
        // then we can accumulate the probs within the child node.
        child = r_node->extend_path(id, ts, prob, token, writer);
        return child;
    }

    // If the current node has no child with the provided id,
    // then we can create a new child.
    child = new Node<T>(id, ts, prob, token, this);

    this->childs.push_back(child);
    writer.push_back(child);
    return child;
}

#endif // _ZCTC_TRIE_H
