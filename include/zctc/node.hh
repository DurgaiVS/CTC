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
	int hotword_length, ts, b_ts, tk_ts, seq_length;
	T _tk_prob, _b_prob, _intrm_score, _squash_prob, _confident_prob;
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
		, _squash_prob(zctc::ZERO)
		, _confident_prob(zctc::ZERO)
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
		if (this->parent == nullptr) {
			this->seq_length = 0;
			return;
		}

		this->p_score = parent->score;
		this->seq_length = parent->seq_length + 1;
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
		, seq_length(other.seq_length)
		, _tk_prob(other._tk_prob)
		, _b_prob(other._b_prob)
		, _intrm_score(other._intrm_score)
		, _squash_prob(other._squash_prob)
		, _confident_prob(other._confident_prob)
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
		, lm_state(other.lm_state) // TODO: Verify if the copy constructor is clean.
		, lexicon_state(other.lexicon_state)
		, hotword_state(other.hotword_state)
	{
	}

	~Node()
	{
		for (Node<T>* child : *this)
			delete child;
	}

	inline void acc_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
	inline void acc_tk_and_parent_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
	T update_score(float penalty, int curr_ts, const float beta, std::vector<Node<T>*>& more_confident_repeats);

	Node<T>* extend_path(int id, int ts, T prob, const std::string token, std::vector<Node<T>*>& writer);
	inline Node<T>* acc_repeat_token_prob(int ts, T prob, std::vector<Node<T>*>& writer);

	// element-wise iterator for this class,
	typename std::vector<Node<T>*>::iterator begin() noexcept { return this->childs.begin(); }
	typename std::vector<Node<T>*>::iterator end() noexcept { return this->childs.end(); }
	typename std::vector<Node<T>*>::const_iterator cbegin() const noexcept { return this->childs.cbegin(); }
	typename std::vector<Node<T>*>::const_iterator cend() const noexcept { return this->childs.cend(); }
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
T
zctc::Node<T>::update_score(float penalty, int curr_ts, const float beta,
							std::vector<zctc::Node<T>*>& more_confident_repeats)
{
	/*
		In case, if a node encounters a blank and a repeat token
		at the same timestep, this could cause the node to have
		two calls to this function, which could lead to the
		node's score to be updated twice, which is not desirable.

		To avoid a check when inserting the nodes into the writer,
		we are going with this approach.
	*/
	if (!this->_update_required)
		return this->h_score;

	if (this->_max_prob > this->max_prob) {
		Node<T>* node = new Node<T>(*this);
		more_confident_repeats.emplace_back(node);

		node->max_prob = node->_max_prob;
		node->ts = curr_ts;
		node->_tk_prob = node->_confident_prob;
		node->_confident_prob = zctc::ZERO;
		node->update_score(penalty, curr_ts, beta, more_confident_repeats);

		/*
		TODO: What if `_tk_prob` included the token's prob
			  two times, like,
				parent->curr_node : as duplicate
				curr_node : as duplicate

			One work around is, subtract the `log(_max_prob)` from
			`_tk_prob` till it is <= `0`. If < `0`, then
			there could possibly be a change in parent score,
			which was included in the `_tk_prob`, so,
			add `log(_max_prob)` to it once...
		*/
		this->_max_prob = this->max_prob;
		this->_squash_prob = zctc::ZERO;
		// this->max_prob = this->_max_prob;
		// this->ts = curr_ts;
	}

	/*
		Here, the `_tk_prob` and `_b_prob` and `lm_prob` will be in
		linear scale, `p_score` and `score` will be in log scale,
	*/
	this->_intrm_score += std::log(this->_tk_prob + this->_b_prob);
	if (this->_squash_prob != zctc::ZERO) {
		this->_intrm_score = std::log(std::exp(this->_intrm_score) + this->_squash_prob);
		this->_squash_prob = zctc::ZERO;
	}

	this->score = this->p_score + this->_intrm_score;
	if (!this->is_lex_path)
		this->score += penalty;

	this->h_score = this->score + this->lm_prob + (beta * this->seq_length);

	if (this->is_hotpath) {
		this->h_score = this->h_score + (this->hotword_length * this->hotword_weight);
	}

	if (this->_tk_prob != zctc::ZERO) {
		this->tk_ts = curr_ts;
		this->_tk_prob = zctc::ZERO;
	}
	if (this->_b_prob != zctc::ZERO) {
		this->b_ts = curr_ts;
		this->_b_prob = zctc::ZERO;
	}

	this->_update_required = false;

	return this->h_score;
}

template <typename T>
void
zctc::Node<T>::acc_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
	/*
	NOTE: Instead of creating a duplicate when we encounter a more
		  confident repeat token, we'll just cache the most confident
		  probability in `_confident_prob` and when creating new node
		  due to confidence timestep update, we'll consider this probs,

					--> blank
					|
				a ---
					|
					--> a(but more confident)

			the above case will be encountered as two different paths,

			1. a -> blank
			2. a -> (blank | a)

			if we accept the token probability, we have to update the
			timestep too, so, we'll just take the blank only for the
			first path and won't consider the token probs,
			and for the second path, we'll consider the token probs
			as well as the blank probs.
	*/
	this->_update_required = true;
	writer.emplace_back(this);

	if (prob > this->max_prob) {
		this->_confident_prob += prob;
		this->_max_prob = prob;
	} else {
		this->_tk_prob += prob;
	}
}

template <typename T>
void
zctc::Node<T>::acc_tk_and_parent_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
	this->_update_required = true;
	writer.emplace_back(this);
	/*
	NOTE: Please look at `acc_prob` functions comment to understand how we are handling
		  duplicate but more confident token.
	*/

	if (this->parent->score == this->p_score) {
		if (prob > this->max_prob) {
			this->_confident_prob += prob;
			this->_max_prob = prob;
		} else {
			this->_tk_prob += prob;
		}
		return;
	}

	if (prob > this->max_prob) {
		this->_max_prob = prob;
	}

	/*
	TODO: Evaluate this case, how can we accumulate the updated
		  probs of parent to the child, as per the expression
		  in the `update_score` function.
	*/
	T diff_prob = this->parent->score - this->p_score;
	this->_squash_prob += std::exp(diff_prob + std::log(prob));
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::acc_repeat_token_prob(int ts, T prob, std::vector<zctc::Node<T>*>& writer)
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
	if (this->tk_ts >= this->b_ts)
		this->acc_prob(prob, writer);

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
			if (r_node->id != id)
				continue;

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
		if (r_node->id != id)
			continue;

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
