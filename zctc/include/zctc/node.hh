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

class Node {
public:
	const bool is_clone, only_prev_b;
	const int id;
	const std::string token;

	bool is_lex_path, is_start_of_word, is_hotpath, is_at_writer, is_deprecated;
	int ts, b_ts, tk_ts;
	double tk_prob, b_prob, prev_b_score, squash_score, prev_score;
	double max_prob, _max_prob, p_score, score, ovrl_score, lm_lex_score;
	double hw_score;

	Node* parent;
	lm::ngram::State lm_state;
	fst::StdVectorFst::StateId lexicon_state, hotword_state;
	std::vector<Node*> childs;
	std::vector<Node*>& alt_childs;

	Node(int id, int ts, double prob, const std::string token, Node* parent, bool only_prev_b = false)
		: is_clone(false)
		, only_prev_b(only_prev_b)
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
		, b_prob(0.0)
		, prev_b_score(0.0)
		, squash_score(0.0)
		, max_prob(prob)
		, _max_prob(prob)
		, p_score(0.0)
		, score(0.0)
		, ovrl_score(0.0)
		, lm_lex_score(0.0)
		, prev_score(0.0)
		, hw_score(0.0)
		, parent(parent)
		, alt_childs(this->childs)
	{
		if (this->parent == nullptr) {
			return;
		}

		this->lm_lex_score = parent->lm_lex_score;

		if (only_prev_b) {
			this->p_score = parent->prev_score + parent->prev_b_score;
			this->score = this->p_score;

			return;
		}

		this->p_score = parent->score;
		this->score = parent->score;
	}

	/**
	 * @brief Clone Constructor
	 *
	 * @note This constructor is used in the case where
	 * 		 `ref` is not the direct child of `parent`
	 * 		 node, but the child of `parent`'s clone
	 * 		 `source` node.
	 */
	Node(int ts, double prob, Node* parent, Node* ref)
		: is_clone(true)
		, only_prev_b(ref->only_prev_b)
		, id(ref->id)
		, token(ref->token)
		, is_lex_path(ref->is_lex_path)
		, is_start_of_word(ref->is_start_of_word)
		, is_hotpath(ref->is_hotpath)
		, is_at_writer(true) // NOTE: Should be inserted after constructor call
		, is_deprecated(false)
		, ts(ref->ts)
		, b_ts(ref->b_ts)
		, tk_ts(ref->tk_ts)
		, tk_prob(ref->tk_prob)
		, b_prob(ref->b_prob)
		, prev_b_score(ref->prev_b_score)
		, squash_score(ref->squash_score)
		, max_prob(ref->max_prob)
		, _max_prob(ref->_max_prob)
		, p_score(ref->p_score)
		, score(ref->score)
		, ovrl_score(ref->ovrl_score)
		, lm_lex_score(ref->lm_lex_score)
		, prev_score(ref->prev_score)
		, hw_score(ref->hw_score)
		, parent(parent)
		, lm_state(ref->lm_state)
		, lexicon_state(ref->lexicon_state)
		, hotword_state(ref->hotword_state)
		, alt_childs(ref->childs)
	{
		ref->is_deprecated = true;
	}

	// Copy Constructor
	Node(Node& other)
		: is_clone(true)
		, only_prev_b(other.only_prev_b)
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
		, tk_prob(other.tk_prob)
		, b_prob(other.b_prob)
		, prev_b_score(other.prev_b_score)
		, squash_score(other.squash_score)
		, max_prob(other.max_prob)
		, _max_prob(other._max_prob)
		, p_score(other.p_score)
		, score(other.score)
		, ovrl_score(other.ovrl_score)
		, lm_lex_score(other.lm_lex_score)
		, prev_score(other.prev_score)
		, hw_score(other.hw_score)
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
		for (Node* child : *this)
			delete child;
	}

	inline void acc_prob(double prob, std::vector<Node*>& writer);
	inline void acc_tk_and_parent_prob(double prob, std::vector<Node*>& writer);
	inline void acc_repeat_token_prob_for_cloned(int ts, double prob, Node* r_node, std::vector<Node*>& writer,
												 std::vector<Node*>& reader);

	double update_score(int curr_ts, std::vector<Node*>& more_confident_repeats);

	inline Node* acc_repeat_token_prob(int ts, double prob, std::vector<Node*>& writer, std::vector<Node*>& reader);

	Node* extend_path(int id, int ts, double prob, const std::string token, std::vector<Node*>& writer,
					  std::vector<Node*>& reader);

	// element-wise iterator for this class,
	typename std::vector<Node*>::iterator begin() noexcept { return this->childs.begin(); }
	typename std::vector<Node*>::iterator end() noexcept { return this->childs.end(); }
	typename std::vector<Node*>::const_iterator cbegin() const noexcept { return this->childs.cbegin(); }
	typename std::vector<Node*>::const_iterator cend() const noexcept { return this->childs.cend(); }
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

/**
 * @brief Exponential sum of two numbers in log scale, and returning the result in log scale.
 *
 * @tparam T The type of the numbers.
 * @param x The first number in log scale.
 * @param y The second number in log scale.
 *
 * @return The result of the exponential sum of the two numbers in log scale.
 */
template <typename T>
inline T
log_sum_exp(T x, T y)
{
	T max_val = std::max(x, y);
	return std::log(std::exp(x - max_val) + std::exp(y - max_val)) + max_val;
}

/**
 * @brief Exponential difference of two numbers in log scale, and returning the result in log scale.
 *
 * @tparam T The type of the numbers.
 * @param x The first number in log scale.
 * @param y The second number in log scale.
 *
 * @return The result of the exponential difference of the two numbers in log scale.
 */
template <typename T>
inline T
log_diff_exp(T x, T y)
{
	T max_val = std::max(x, y);
	return std::log(std::exp(x - max_val) - std::exp(y - max_val)) + max_val;
}

/**
 * @brief Updates the score of the node in current timestep and returns the updated score.
 * 		  The score is updated based on the token and blank probabilities, and the
 * 		  previous overlapping extension from the parent node. This function should
 * 		  be called only once per timestep and also at the end of parsing the timestep
 * 		  logits.
 *
 * @param curr_ts The current timestep.
 * @param more_confident_repeats The vector to store the more confident repeat tokens, since
 * 								 the more confident repeat nodes were created here to take
 * 								 into account all possible ways of arriving probabilities
 * 								 by the old node.
 *
 * @return The updated score of the node.
 */
double
zctc::Node::update_score(int curr_ts, std::vector<zctc::Node*>& more_confident_repeats)
{
	/**
	 * NOTE: In case, if a node encounters a blank and a repeat token
	 * 		 at the same timestep, this could cause the node to have
	 * 		 two calls to this function, which could lead to the
	 * 		 node's score to be updated twice, which is not desirable.
	 *
	 * 		 To avoid a check when inserting the nodes into the writer,
	 * 		 we are going with this approach.
	 */
	if (this->_max_prob > this->max_prob) {

		if (this->childs.size() != 0) {
			/**
			 * NOTE: This is a more confident repeat token,
			 * 		 so, we'll create a new node and update the
			 * 		 score with the most confident probability.
			 */
			Node* node = new Node(*this);
			more_confident_repeats.emplace_back(node);

			node->tk_prob = node->_max_prob;
			node->max_prob = node->_max_prob;
			node->ts = curr_ts;

			this->_max_prob = this->max_prob;
			this->squash_score = 0.0;
			this->is_at_writer = false;
			this->is_deprecated = true;

			return node->update_score(curr_ts, more_confident_repeats);
		}

		this->tk_prob = this->_max_prob;
		this->max_prob = this->_max_prob;
		this->ts = curr_ts;
	}

	/**
	 * NOTE: Here,
	 * 		 `*_prob` will be in linear scale,
	 * 		 `*_score, ` will be in log scale,
	 */
	double prev_score = this->score;
	this->score = prev_score + std::log(this->tk_prob + this->b_prob);

	if ((this->prev_b_score != 0.0) && this->tk_prob != 0.0) {
		this->score = log_diff_exp(this->score, this->prev_score + this->prev_b_score + std::log(this->tk_prob));
	}
	if (this->squash_score != 0.0) {
		this->score = log_sum_exp(this->score, this->squash_score);
		this->squash_score = 0.0;
	}

	this->ovrl_score = this->score + this->lm_lex_score + this->hw_score;
	this->prev_score = prev_score;

	if (this->tk_prob != 0.0) {
		this->tk_ts = curr_ts;
		this->tk_prob = 0.0;
	}

	if (this->b_prob != 0.0) {
		this->b_ts = curr_ts;
		this->prev_b_score = std::log(this->b_prob);
		this->b_prob = 0.0;
	} else {
		this->prev_b_score = 0.0;
	}

	this->is_at_writer = false;
	return this->ovrl_score;
}

/**
 * @brief Accumulates the token probability to the node, and if the probability
 * 		  is more confident than the node's probability, then this value is cached
 * 		  in `_max_prob` and updated during `update_score` function.
 *
 * @param prob The token probability to be accumulated.
 * @param writer The vector to store the nodes to be written to the next timestep.
 *
 * @return void
 */
void
zctc::Node::acc_prob(double prob, std::vector<zctc::Node*>& writer)
{
	/**
	 * NOTE: Instead of creating a duplicate when we encounter a more
	 * 		 confident repeat token, we'll just cache the most confident
	 * 		 probability in `_max_prob` and when creating new node
	 * 		 due to confidence timestep update, we'll consider this probs,
	 *
	 * 			--> blank
	 * 			|
	 * parent --->	a ---> some other tokens
	 * 			|
	 * 			--> a(but more confident)
	 *
	 * 		 the above case will be encountered as two different paths,
	 *
	 * 		 1. a -> some childs
	 * 		 2. a -> (blank | a)
	 *
	 * 		 if we accept the token probability, we have to update the
	 * 		 timestep too, so, we won't consider the token and blank
	 * 		 probs for the first path, so the child nodes won't have
	 * 		 a mess in the timestep order, and for the second path,
	 * 		 we'll consider both probs (if provided) as well as the
	 * 		 blank probs.
	 */
	if (!this->is_at_writer) {
		writer.emplace_back(this);
		this->is_at_writer = true;
	}

	if (prob > this->max_prob) {
		this->_max_prob = prob;
	}

	this->tk_prob = prob;
}

/**
 * @brief Accumulates the token probability as well as the updated parent probability
 * 		  to the node, and if the probability is more confident than the node's probability,
 * 		  then this value is cached in `_max_prob` and updated during `update_score` function.
 *
 * @param prob The token probability to be accumulated.
 * @param writer The vector to store the nodes to be written to the next timestep.
 *
 * @return void
 */
void
zctc::Node::acc_tk_and_parent_prob(double prob, std::vector<zctc::Node*>& writer)
{
	if (!this->is_at_writer) {
		writer.emplace_back(this);
		this->is_at_writer = true;
	}
	/**
	 * NOTE: Please look at `acc_prob` functions comment to understand
	 * 		 how we are handling duplicate but more confident token.
	 */
	if (prob > this->max_prob) {
		this->_max_prob = prob;
	}

	if ((!this->only_prev_b) && (this->parent->score == this->p_score)) {
		this->tk_prob = prob;

	} else if (this->only_prev_b) {
		double p_score = this->parent->prev_score + this->parent->prev_b_score;

		if (this->p_score != p_score) {
			this->p_score = p_score;
			this->squash_score = p_score + std::log(prob);
		} else {
			this->tk_prob = prob;
		}

	} else {
		this->p_score = this->parent->score;
		this->squash_score = this->parent->score + std::log(prob);
	}
}

/**
 * @brief Accumulates the repeat token probability for cloned node, if the
 * 		  reference node is not the direct child of `this` node, but the child
 * 		  of the clone `source` node.
 *
 * @param ts The timestep of the token.
 * @param prob The token probability to be accumulated.
 * @param r_node The reference node.
 * @param writer The vector to store the nodes to be written to the next timestep.
 * @param reader The vector to remove the redundant node existence in case of cloning.
 *
 * @return void
 */
void
zctc::Node::acc_repeat_token_prob_for_cloned(int ts, double prob, zctc::Node* r_node, std::vector<zctc::Node*>& writer,
											 std::vector<zctc::Node*>& reader)
{

	zctc::Node* child;
	if (r_node->childs.size() == 0) {
		// Delete this r_node ref from the alt_childs list.
		child = r_node;
		child->parent = this;
		if (!child->is_at_writer) {
			writer.emplace_back(child);
			child->is_at_writer = true;
		}

		std::iter_swap(std::find(this->alt_childs.begin(), this->alt_childs.end(), r_node), this->alt_childs.end() - 1);
		this->alt_childs.erase(this->alt_childs.end() - 1);
	} else {
		child = new zctc::Node(ts, prob, this, r_node);
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
	} else if (prob > child->max_prob) {
		child->_max_prob = prob;
	}

	this->childs.emplace_back(child);
}

/**
 * @brief Accumulates the repeat token probability to the node, and based on the
 * 		  last encounterence of the token and blank, whether the path should be
 * 		  extended or not is decided.
 *
 * @param ts The timestep of the token.
 * @param prob The token probability to be accumulated.
 * @param writer The vector to store the nodes to be written to the next timestep.
 * @param reader The vector to remove the redundant node existence in case of cloning.
 *
 * @return The child node if the path is extended, else `nullptr`.
 */
zctc::Node*
zctc::Node::acc_repeat_token_prob(int ts, double prob, std::vector<zctc::Node*>& writer,
								  std::vector<zctc::Node*>& reader)
{
	/**
	 * NOTE: In case, if the token is the most recent than the blank, or,
	 * 		 if the blank and token are at the same timestep,
	 * 		 within this node,
	 * 		 we can accumulate the probs with the current node.
	 *
	 * 		 The assumption here is,
	 * 		 If the token is the most recent than the blank, then we will
	 * 		 treat the token as monotonic, and we won't extend the path,
	 * 		 rather we will accumulate the probs with the current node.
	 */
	if (this->tk_ts >= this->b_ts)
		this->acc_prob(prob, writer);

	/**
	 * NOTE: In case, if the blank is more recent than the token, or,
	 * 		 if the blank and token are at the same timestep,
	 * 		 within this node,
	 * 		 we can create a new child, assuming that the current token
	 * 		 (the one passed in the argument) is preceded by a blank.
	 *
	 * 		 The assumption here is,
	 * 		 If the blank is the most recent than the token, then we will
	 * 		 treat the token as non-monotonic, and we will extend the path,
	 * 		 rather than accumulating the probs with the current node, due to
	 * 		 the fact that the token is preceded by a blank.
	 */
	if (this->b_ts >= this->tk_ts) {
		/**
		 * NOTE: In case, if the child is already available, we can just update
		 * 		 the probs and return the child.
		 * 		 Not sure if this case is possible or not, but just wanted to
		 * 		 ensure that we are not creating duplicate child nodes.
		 */
		for (zctc::Node* r_node : *this) {
			if ((r_node->id != id) || r_node->is_deprecated)
				continue;

			r_node->acc_tk_and_parent_prob(prob, writer);
			return nullptr;
		}

		if (this->is_clone) {
			/**
			 * NOTE: If this is a cloned node, then we'll look
			 * 		 for the `source` node's child list too.
			 */
			for (Node* r_node : this->alt_childs) {
				if ((r_node->id != id) || r_node->is_deprecated)
					continue;

				this->acc_repeat_token_prob_for_cloned(ts, prob, r_node, writer, reader);
				return nullptr;
			}
		}

		/**
		 * NOTE: If the child is not available, then we can create a new child.
		 * 		 NOTE, here we are using the `only_prev_prob` flag, to indicate
		 * 		 that the extended node is preceeded by a blank, so if the current
		 * 		 node has both `blank` and `token` encountered previously, then
		 * 		 we'll only consider the previous `blank`.
		 */
		zctc::Node* child = new zctc::Node(id, ts, prob, token, this, true);

		this->childs.emplace_back(child);
		writer.emplace_back(child);
		child->is_at_writer = true;

		return child;
	}

	return nullptr;
}

/**
 * @brief Extends the path with the provided id, and accumulates the token probability
 * 		  to the node, in case of non-repeat token, and in case of repeat token, the
 * 		  probs are accumulated within the child node, and the parent probs are updated
 * 		  in the child node too. And based on the last encounterence of the token and blank,
 * 		  whether the path should be extended or not is decided.
 *
 * @param id The id of the token to be extended.
 * @param ts The timestep of the token.
 * @param prob The token probability to be accumulated.
 * @param token The token string.
 * @param writer The vector to store the nodes to be written to the next timestep.
 * @param reader The vector to remove the redundant node existence in case of cloning.
 *
 * @return The child node if the path is extended, else `nullptr`.
 */
zctc::Node*
zctc::Node::extend_path(int id, int ts, double prob, const std::string token, std::vector<zctc::Node*>& writer,
						std::vector<zctc::Node*>& reader)
{
	if (id == this->id)
		return this->acc_repeat_token_prob(ts, prob, writer, reader);

	for (Node* r_node : *this) {
		if ((r_node->id != id) || r_node->is_deprecated)
			continue;

		/**
		 * NOTE: If the current node has a child with the provided id,
		 * 		 then we can accumulate the probs within the child node.
		 *
		 * 		 Since, we're accumulating the repeated and blank probs
		 * 		 within the node itself, there could be a possibility that
		 * 		 the parent's probs have changed from the timestep this
		 * 		 child was created, so we need to update the parent probs
		 * 		 value in the child node too...More details in the function
		 * 		 definition.
		 */
		r_node->acc_tk_and_parent_prob(prob, writer);
		return nullptr;
	}

	if (this->is_clone) {
		/**
		 * NOTE: If this is a cloned node, then we'll look
		 * 		 for the `source` node's child list too.
		 */
		for (Node* r_node : this->alt_childs) {
			if ((r_node->id != id) || r_node->is_deprecated)
				continue;

			this->acc_repeat_token_prob_for_cloned(ts, prob, r_node, writer, reader);
			return nullptr;
		}
	}

	/**
	 * NOTE: If the current node has no child with the provided id,
	 * 		 then we can create a new child node and extend the path.
	 */
	zctc::Node* child = new zctc::Node(id, ts, prob, token, this);

	this->childs.emplace_back(child);
	writer.emplace_back(child);
	child->is_at_writer = true;

	return child;
}

#endif // _ZCTC_TRIE_H
