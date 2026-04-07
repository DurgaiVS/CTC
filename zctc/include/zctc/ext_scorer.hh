#ifndef _ZCTC_EXT_SCORER_H
#define _ZCTC_EXT_SCORER_H

#include "fst/fstlib.h"
#include "lm/model.hh"

#include "./node.hh"

namespace zctc {

class ExternalScorer {
public:
	const bool enabled;
	const char tok_sep;
	const int apostrophe_id;
	const float alpha, beta, lex_penalty;
	lm::base::Model* lm;
	fst::StdVectorFst* lexicon;

	ExternalScorer(char tok_sep, int apostrophe_id, float alpha, float beta, float lex_penalty, char* lm_path,
				   char* lexicon_path)
		: enabled(lm_path || lexicon_path)
		, tok_sep(tok_sep)
		, apostrophe_id(apostrophe_id)
		, alpha(alpha)
		, beta(beta)
		, lex_penalty(lex_penalty)
		, lm(nullptr)
		, lexicon(nullptr)
	{

		if (lm_path)
			this->lm = lm::ngram::LoadVirtual(lm_path);

		if (lexicon_path)
			this->lexicon = fst::StdVectorFst::Read(lexicon_path);
	}

	~ExternalScorer()
	{
		if (this->lm)
			delete this->lm;

		if (this->lexicon)
			delete this->lexicon;
	}

	template <typename T>
	inline void start_of_word_check(zctc::Node<T>* node, fst::StdVectorFst* hotwords_fst) const;
	template <typename T>
	inline void initialise_start_states(zctc::Node<T>* root, fst::StdVectorFst* hotwords_fst) const;

	int shortest_eos_from(fst::StdVectorFst* hotwords_fst, fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher,
						  fst::StdVectorFst::StateId state) const;
	template <typename T>
	void run_ext_scoring(zctc::Node<T>* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
						 fst::StdVectorFst* hotwords_fst,
						 fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const;
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

/**
 * @brief Check whether the provided node is a start of word or not, and
 * 		  initialise the lexicon and hotword states for the node, if so.
 * 		  For BPE tokenized vocab, the start of word is considered as
 * 		  the token which is not a subword token, and not an apostrophe
 * 		  token, and not child of an apostrophe token.
 *
 * @param node The node for which the start of word check is to be done.
 * @param hotwords_fst Initialise the start of word hotword state for the node from this FST.
 *
 * @return void
 */
template <typename T>
void
zctc::ExternalScorer::start_of_word_check(zctc::Node<T>* node, fst::StdVectorFst* hotwords_fst) const
{
	node->is_start_of_word = !(node->id == this->apostrophe_id || node->parent->id == this->apostrophe_id
							   || node->token.at(0) == this->tok_sep);

	if (!node->is_start_of_word)
		return;

	if (this->lexicon)
		node->lexicon_state = this->lexicon->Start();

	if (hotwords_fst)
		node->hotword_state = hotwords_fst->Start();
}

/**
 * @brief Initialise the start states for the provided node, for the lexicon,
 * 		  language model and hotwords FSTs.
 *
 * @param root The node for which the start states are to be initialised.
 * @param hotwords_fst Initialise the start of word hotword state for the node from this FST.
 *
 * @return void
 */
template <typename T>
void
zctc::ExternalScorer::initialise_start_states(zctc::Node<T>* root, fst::StdVectorFst* hotwords_fst) const
{
	if (this->lexicon)
		root->lexicon_state = this->lexicon->Start();

	if (this->lm)
		this->lm->BeginSentenceWrite(&(root->lm_state));

	if (hotwords_fst)
		root->hotword_state = hotwords_fst->Start();
}

/**
 * @brief Calculates the shortest distance to the end of sentence from the provided state in the provided FST.
 *
 * @param fst The FST to be used for the search.
 * @param state The state from which the shortest distance to the end of sentence is to be calculated.
 *
 * @return int The shortest distance to the end of sentence from the provided state in the provided FST.
 */
int
zctc::ExternalScorer::shortest_eos_from(fst::StdVectorFst* hotwords_fst,
										fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher,
										fst::StdVectorFst::StateId state) const
{
	bool final = false;
	int hop_count = 0;

	for (; !final; hop_count++) {
		auto final_w = hotwords_fst->Final(state);
		if (final_w != fst::StdArc::Weight::Zero()) {
			final = true;
			continue;
		}

		hotwords_matcher->SetState(state);
		hotwords_matcher->Next();
		state = hotwords_matcher->state_;
	}
	return hop_count;
}

/**
 * @brief Run the external scoring for the provided node, considering the
 * 		  language model, lexicon, hotwords FSTs and beta word penalty
 * 		  using the external scorer parameters.
 *
 * @param node The node for which the external scoring is to be done.
 * @param lexicon_matcher The lexicon matcher to be used for lexicon searching.
 * @param hotwords_fst The hotwords FST to be used for hotword scoring.
 * @param hotwords_matcher The hotwords matcher to be used for hotword searching.
 *
 * @return void
 */
template <typename T>
void
zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
									  fst::StdVectorFst* hotwords_fst,
									  fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const
{

	if (this->lm) {

		lm::WordIndex word_id = this->lm->BaseVocabulary().Index(node->token);

		if (word_id == this->lm->BaseVocabulary().NotFound()) {
			node->lm_lex_score += -1000; // OOV char
		} else {
			/**
			 * NOTE: Since KenLM returns the log probability with base 10,
			 * 		 converting the log probability to base e, using,
			 *
			 * 		 logb(x) = loga(x) / loga(b)
			 */
			node->lm_lex_score
				+= (this->alpha
					* (this->lm->BaseScore(&(node->parent->lm_state), word_id, &(node->lm_state)) / zctc::LOG_A_OF_B))
				   + this->beta;
		}
	}

	this->start_of_word_check(node, hotwords_fst);

	/**
	 * NOTE: Hotword scores and beta word penalty were accumulated in seperate variable
	 * 		 because, these values will not be passed hereditarily to the child nodes.
	 *
	 * 		 But, the language model and lexicon scores will be passed to the child nodes.
	 */
	if (hotwords_fst && (node->parent->is_hotpath || node->is_start_of_word)) {
		/**
		 * NOTE: If the node is the start of word, then
		 * 		 check whether the parent is a hotword path or not.
		 * 		 If yes, then continue from the parent's hotword state.
		 * 		 If not, then start from the initial state of the hotwords FST.
		 */
		fst::StdVectorFst::StateId state
			= (node->is_start_of_word && (!node->is_hotpath)) ? node->hotword_state : node->parent->hotword_state;
		hotwords_matcher->SetState(state);

		if (hotwords_matcher->Find(node->id)) {
			const fst::StdArc& arc = hotwords_matcher->Value();
			/**
			 * NOTE: Here,
			 * 		 arc.olabel is the token length so far in the hotword,
			 * 		 arc.weight.Value() is the weight for each hotword token.
			 */
			node->hotword_state = arc.nextstate;
			int remaining_hopcount = this->shortest_eos_from(hotwords_fst, hotwords_matcher, node->hotword_state);
			node->hw_score = zctc::quadratic_hw_score(arc.olabel, arc.olabel + remaining_hopcount, arc.weight.Value());
			node->is_hotpath = true;

		} else if (node->is_start_of_word) {
			hotwords_matcher->SetState(node->hotword_state);
			if (hotwords_matcher->Find(node->id)) {
				const fst::StdArc& arc = hotwords_matcher->Value();
				node->hotword_state = arc.nextstate;
				int remaining_hopcount = this->shortest_eos_from(hotwords_fst, hotwords_matcher, node->hotword_state);
				node->hw_score
					= zctc::quadratic_hw_score(arc.olabel, arc.olabel + remaining_hopcount, arc.weight.Value());
				node->is_hotpath = true;
			}
		}

		if (node->parent->is_hotpath
			&& (hotwords_fst->Final(node->parent->hotword_state) != fst::StdArc::Weight::Zero())
			&& (hotwords_matcher->state_ == hotwords_fst->Start())) {
			/**
			 * NOTE: Adding the previously completed hotword score to the `lm_lex_score` as this
			 * 		 attribute's value will be passed hereditarily to the successor nodes.
			 */
			node->lm_lex_score += node->parent->hw_score;
		}
	}

	if (this->lexicon) {
		/**
		 * NOTE: Improper combinations of tokens are penalized with `lex_penalty`.
		 */
		if (!(node->parent->is_lex_path || node->is_start_of_word)) {

			node->is_lex_path = false;
			node->lm_lex_score += (node->is_hotpath ? 0 : this->lex_penalty);

		} else {
			/**
			 * NOTE: If the node is the start of word, then
			 * 		 check whether the parent is a lexicon path or not.
			 * 		 If yes, then continue from the parent's lexicon state.
			 * 		 If not, then start from the initial state of the lexicon FST.
			 */
			fst::StdVectorFst::StateId state = (node->is_start_of_word && (!node->parent->is_lex_path))
												   ? node->lexicon_state
												   : node->parent->lexicon_state;
			lexicon_matcher->SetState(state);

			/**
			 * NOTE: If the node's parent is a valid lexicon path, and also the
			 * 		 node is a start of the word, then we'll first check if the
			 * 		 node is a proper lexicon child for the parent, if not, then
			 * 		 we'll check if the start of the word is a seperate
			 * 		 lexicon entity.
			 */
			if (lexicon_matcher->Find(node->id)) {
				node->lexicon_state = lexicon_matcher->Value().nextstate;
				node->is_lex_path = true;

			} else if (node->is_start_of_word && node->parent->is_lex_path) {
				lexicon_matcher->SetState(node->lexicon_state);
				if (lexicon_matcher->Find(node->id)) {
					node->lexicon_state = lexicon_matcher->Value().nextstate;
					node->is_lex_path = true;
				} else {
					node->is_lex_path = false;
					node->lm_lex_score += (node->is_hotpath ? 0 : this->lex_penalty);
				}

			} else {
				node->is_lex_path = false;
				node->lm_lex_score += (node->is_hotpath ? 0 : this->lex_penalty);
			}
		}
	}
}

#endif // _ZCTC_EXT_SCORER_H
