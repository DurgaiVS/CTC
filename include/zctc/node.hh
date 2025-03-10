#ifndef _ZCTC_TRIE_H
#define _ZCTC_TRIE_H

#include <algorithm>
#include <cassert>
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
	const bool is_clone, only_prev_blank;
	const int id;
	const std::string token;

	bool is_lex_path, is_start_of_word, is_hotpath, is_at_writer, is_deprecated;
	int ts, b_ts, tk_ts, seq_length;
	T tk_prob, b_prob, squash_score, more_confident_prob, prev_b_prob;
	T max_prob, _max_prob, p_score, _p_score, score, h_score, ext_score;
	T intrm_score, aft_intrm_score;

	Node<T>* parent;
	lm::ngram::State lm_state;
	fst::StdVectorFst::StateId lexicon_state, hotword_state;
	std::vector<Node<T>*> childs;
	std::vector<Node<T>*>& alt_childs;

	Node(int id, int ts, T prob, const std::string token, Node<T>* parent, bool only_prev_blank = false)
		: is_clone(false)
		, only_prev_blank(only_prev_blank)
		, id(id)
		, token(token)
		, is_lex_path(true)
		, is_start_of_word(false)
		, is_hotpath(false)
		, is_at_writer(false)
		, is_deprecated(false)
		, ts(ts)
		, b_ts(-1)
		, tk_ts(ts)
		, tk_prob(prob)
		, b_prob(zctc::ZERO)
		, squash_score(zctc::ZERO)
		, more_confident_prob(zctc::ZERO)
		, prev_b_prob(zctc::ZERO)
		, max_prob(prob)
		, _max_prob(prob)
		, p_score(zctc::ZERO)
		, _p_score(zctc::ZERO)
		, score(zctc::ZERO)
		, h_score(zctc::ZERO)
		, ext_score(zctc::ZERO)
		, intrm_score(zctc::ZERO)
		, aft_intrm_score(zctc::ZERO)
		, parent(parent)
		, alt_childs(this->childs)
	{
		if (this->parent == nullptr) {
			this->seq_length = 0;
			return;
		}

		this->seq_length = parent->seq_length + 1;
		if (!only_prev_blank) {
			this->p_score = parent->score;
			this->_p_score = parent->score;
			return;
		}
		this->p_score = parent->_p_score + parent->intrm_score + std::log(parent->prev_b_prob);
		this->_p_score = this->p_score;
	}

	// Clone Constructor
	Node(int ts, T prob, Node<T>* parent, Node<T>* ref)
		: is_clone(true)
		, only_prev_blank(ref->only_prev_blank)
		, id(ref->id)
		, token(ref->token)
		, is_lex_path(ref->is_lex_path)
		, is_start_of_word(ref->is_start_of_word)
		, is_hotpath(ref->is_hotpath)
		, is_at_writer(true) // NOTE: Should be replaced after constructor call
		, is_deprecated(false)
		, ts(ref->ts)
		, b_ts(ref->b_ts)
		, tk_ts(ref->tk_ts)
		, seq_length(ref->seq_length)
		, tk_prob(ref->tk_prob)
		, b_prob(ref->b_prob)
		, squash_score(ref->squash_score)
		, more_confident_prob(ref->more_confident_prob)
		, prev_b_prob(ref->prev_b_prob)
		, max_prob(ref->max_prob)
		, _max_prob(ref->_max_prob)
		, p_score(ref->p_score)
		, _p_score(ref->_p_score)
		, score(ref->score)
		, h_score(ref->h_score)
		, ext_score(ref->ext_score)
		, intrm_score(ref->intrm_score)
		, aft_intrm_score(ref->aft_intrm_score)
		, parent(parent)
		, alt_childs(ref->childs)
	{
		ref->is_deprecated = true;
	}

	// Copy Constructor
	Node(Node& other)
		: is_clone(true)
		, only_prev_blank(other.only_prev_blank)
		, id(other.id)
		, token(other.token)
		, is_lex_path(other.is_lex_path)
		, is_start_of_word(other.is_start_of_word)
		, is_hotpath(other.is_hotpath)
		, is_at_writer(other.is_at_writer)
		, is_deprecated(false)
		, ts(other.ts)
		, b_ts(other.b_ts)
		, tk_ts(other.tk_ts)
		, seq_length(other.seq_length)
		, tk_prob(other.tk_prob)
		, b_prob(other.b_prob)
		, squash_score(other.squash_score)
		, more_confident_prob(other.more_confident_prob)
		, prev_b_prob(other.prev_b_prob)
		, max_prob(other.max_prob)
		, _max_prob(other._max_prob)
		, p_score(other.p_score)
		, _p_score(other._p_score)
		, score(other.score)
		, h_score(other.h_score)
		, ext_score(other.ext_score)
		, intrm_score(other.intrm_score)
		, aft_intrm_score(other.aft_intrm_score)
		, parent(other.parent)
		, lm_state(other.lm_state) // TODO: Verify if the copy constructor is clean.
		, lexicon_state(other.lexicon_state)
		, hotword_state(other.hotword_state)
		, alt_childs(other.childs)
	{
		this->parent->childs.emplace_back(this);
	}

	~Node()
	{
		for (Node<T>* child : *this)
			delete child;
	}

	inline void acc_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
	inline void acc_tk_and_parent_prob(T prob, std::vector<Node<T>*>& writer) noexcept;
	T update_score(int curr_ts, std::vector<Node<T>*>& more_confident_repeats);

	Node<T>* extend_path(int id, int ts, T prob, const std::string token, std::vector<Node<T>*>& writer,
						 std::vector<Node<T>*>& reader);
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
zctc::Node<T>::update_score(int curr_ts, std::vector<zctc::Node<T>*>& more_confident_repeats)
{
	/*
		In case, if a node encounters a blank and a repeat token
		at the same timestep, this could cause the node to have
		two calls to this function, which could lead to the
		node's score to be updated twice, which is not desirable.

		To avoid a check when inserting the nodes into the writer,
		we are going with this approach.
	*/

	assert(this->is_at_writer);
	if (this->more_confident_prob != zctc::ZERO) {
		Node<T>* node = new Node<T>(*this);
		more_confident_repeats.emplace_back(node);

		if (node->_max_prob > node->max_prob) {
			node->max_prob = node->_max_prob;
			node->ts = curr_ts;
		}
		node->tk_prob += node->more_confident_prob;
		node->more_confident_prob = zctc::ZERO;

		this->_max_prob = this->max_prob;
		this->squash_score = zctc::ZERO;
		this->is_at_writer = false;
		this->is_deprecated = true;

		return node->update_score(curr_ts, more_confident_repeats);

		/*
		TODO: What if `tk_prob` included the token's prob
			  two times, like,
				parent->curr_node : as duplicate
				curr_node : as duplicate

			One work around is, subtract the `log(_max_prob)` from
			`tk_prob` till it is <= `0`. If < `0`, then
			there could possibly be a change in parent score,
			which was included in the `tk_prob`, so,
			add `log(_max_prob)` to it once...
		*/

		// this->max_prob = this->_max_prob;
		// this->ts = curr_ts;
	}

	/*
		Here,
		`*_prob` will be in linear scale,
		`*_score, ` will be in log scale,
	*/
	this->intrm_score += this->aft_intrm_score;
	if (this->squash_score != zctc::ZERO) {
		this->aft_intrm_score
			= std::log(this->tk_prob + this->b_prob + std::exp(this->squash_score - this->intrm_score));
		this->squash_score = zctc::ZERO;
	} else {
		this->aft_intrm_score = std::log(this->tk_prob + this->b_prob);
	}

	this->score = this->_p_score + this->intrm_score + this->aft_intrm_score;
	this->h_score = this->score + this->ext_score;

	if (this->tk_prob != zctc::ZERO) {
		this->tk_ts = curr_ts;
		this->tk_prob = zctc::ZERO;
	}
	if (this->b_prob != zctc::ZERO) {
		this->b_ts = curr_ts;
		this->prev_b_prob = this->b_prob;
		this->b_prob = zctc::ZERO;
	}

	this->is_at_writer = false;
	return this->h_score;
}

template <typename T>
void
zctc::Node<T>::acc_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
	/*
	NOTE: Instead of creating a duplicate when we encounter a more
		  confident repeat token, we'll just cache the most confident
		  probability in `more_confident_prob` and when creating new node
		  due to confidence timestep update, we'll consider this probs,

						--> blank
						|
			parent --->	a ---> some other tokens
						|
						--> a(but more confident)

			the above case will be encountered as two different paths,

			1. a -> some childs
			2. a -> (blank | a)

			if we accept the token probability, we have to update the
			timestep too, so, we won't consider the token and blank
			probs for the first path, so the child nodes won't have
			a mess in the timestep order, and for the second path,
			we'll consider both probs (if provided) as well as the
			blank probs.
	*/
	if (!this->is_at_writer) {
		writer.emplace_back(this);
		this->is_at_writer = true;
	}

	if (prob > this->max_prob) {
		this->more_confident_prob += prob;
		this->_max_prob = prob;
	} else {
		this->tk_prob += prob;
	}
}

template <typename T>
void
zctc::Node<T>::acc_tk_and_parent_prob(T prob, std::vector<zctc::Node<T>*>& writer) noexcept
{
	if (!this->is_at_writer) {
		writer.emplace_back(this);
		this->is_at_writer = true;
	}
	/*
	NOTE: Please look at `acc_prob` functions comment to understand how we are handling
		  duplicate but more confident token.
	*/

	if ((this->parent->score == this->p_score) || this->only_prev_blank) {
		if (prob > this->max_prob) {
			this->more_confident_prob += prob;
			this->_max_prob = prob;
		} else {
			this->tk_prob += prob;
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
	assert(this->squash_score == zctc::ZERO);
	T diff_prob = this->parent->score - this->_p_score;
	this->p_score = this->parent->score;
	this->squash_score = diff_prob + std::log(prob);
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
	if ((this->b_ts >= this->tk_ts) && (this->intrm_score != zctc::ZERO)) {
		/*
			In case, if the child is already available, we can just update
			the probs and return the child.
			Not sure if this case is possible or not, but just wanted to
			ensure that we are not creating duplicate child nodes.
		*/
		for (zctc::Node<T>* r_node : *this) {
			if ((r_node->id != id) || r_node->is_deprecated)
				continue;

			r_node->acc_prob(prob, writer);
			return nullptr;
		}

		// If the child is not available, then we can create a new child.
		zctc::Node<T>* child = new zctc::Node<T>(id, ts, prob, token, this, true);

		this->childs.emplace_back(child);
		writer.emplace_back(child);
		child->is_at_writer = true;

		return child;
	}

	return nullptr;
}

template <typename T>
zctc::Node<T>*
zctc::Node<T>::extend_path(int id, int ts, T prob, const std::string token, std::vector<zctc::Node<T>*>& writer,
						   std::vector<zctc::Node<T>*>& reader)
{
	/*
		TODO: In case of repeat, but with most confident prob,
		then, try to create a new node, coz updating in this node
		will affect the previously added child's timesteps.
	*/
	zctc::Node<T>* child;
	if (id == this->id)
		return this->acc_repeat_token_prob(ts, prob, writer);

	for (Node<T>* r_node : *this) {
		if ((r_node->id != id) || r_node->is_deprecated)
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

	if (this->is_clone) {
		/*
		NOTE: If this is a cloned node, then we'll look
			  for the original node's child list too.
		*/
		for (Node<T>* r_node : this->alt_childs) {
			if ((r_node->id != id) || r_node->is_deprecated)
				continue;

			if (r_node->childs.size() == 0) {
				// Delete this r_node ref from the alt_childs list.
				child = r_node;
				child->parent = this;
				if (!child->is_at_writer) {
					writer.emplace_back(child);
					child->is_at_writer = true;
				}

				std::iter_swap(std::find(this->alt_childs.begin(), this->alt_childs.end(), r_node),
							   this->alt_childs.end() - 1);
				this->alt_childs.erase(this->alt_childs.end() - 1);
			} else {
				child = new zctc::Node<T>(ts, prob, this, r_node);
				std::replace(reader.begin(), reader.end(), r_node, child);
				if (r_node->is_at_writer) {
					std::replace(writer.begin(), writer.end(), r_node, child);
				} else {
					writer.emplace_back(child);
					child->is_at_writer = true;
				}
			}

			child->acc_tk_and_parent_prob(prob, writer);
			if (child->ts <= this->ts) {
				child->ts = ts;
				child->tk_ts = ts;
				child->max_prob = prob;
				child->_max_prob = prob;
			}

			this->childs.emplace_back(child);
			return nullptr;
		}
	}

	// If the current node has no child with the provided id,
	// then we can create a new child.
	child = new zctc::Node<T>(id, ts, prob, token, this);

	this->childs.emplace_back(child);
	writer.emplace_back(child);
	child->is_at_writer = true;

	return child;
}

#endif // _ZCTC_TRIE_H
