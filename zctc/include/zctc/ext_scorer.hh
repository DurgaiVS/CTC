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
	const double alpha, beta, lex_penalty;
	lm::base::Model* lm;
	fst::StdVectorFst* lexicon;

	ExternalScorer(char tok_sep, int apostrophe_id, double alpha, double beta, double lex_penalty, char* lm_path,
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

	inline void start_of_word_check(Node* node, fst::StdVectorFst* hotwords_fst) const;
	inline void initialise_start_states(Node* root, fst::StdVectorFst* hotwords_fst) const;

	void run_ext_scoring(zctc::Node* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
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
void
zctc::ExternalScorer::start_of_word_check(zctc::Node* node, fst::StdVectorFst* hotwords_fst) const
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
void
zctc::ExternalScorer::initialise_start_states(zctc::Node* root, fst::StdVectorFst* hotwords_fst) const
{
	if (this->lexicon)
		root->lexicon_state = this->lexicon->Start();

	if (this->lm)
		this->lm->BeginSentenceWrite(&(root->lm_state));

	if (hotwords_fst)
		root->hotword_state = hotwords_fst->Start();
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
void
zctc::ExternalScorer::run_ext_scoring(zctc::Node* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
									  fst::StdVectorFst* hotwords_fst,
									  fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const
{

	if (this->lm) {

		lm::WordIndex word_id = this->lm->BaseVocabulary().Index(node->token);

		if (word_id == this->lm->BaseVocabulary().NotFound()) {
			node->lm_lex_score += -1000; // OOV char
		} else {
			/*
			NOTE: Since KenLM returns the log probability with base 10,
				converting the log probability to base e, using,

				logb(x) = loga(x) / loga(b)
			*/
			node->lm_lex_score
				+= (this->alpha
					* (this->lm->BaseScore(&(node->parent->lm_state), word_id, &(node->lm_state)) / zctc::LOG_A_OF_B))
				   + this->beta;
		}
	}

	this->start_of_word_check(node, hotwords_fst);

	if (this->lexicon) {
		/*
		NOTE: Improper combinations of tokens are penalized with `lex_penalty`.
		*/

		if (!(node->parent->is_lex_path || node->is_start_of_word)) {

			node->is_lex_path = false;
			node->lm_lex_score += this->lex_penalty;

		} else {

			fst::StdVectorFst::StateId state
				= node->is_start_of_word ? node->lexicon_state : node->parent->lexicon_state;
			lexicon_matcher->SetState(state);

			if (lexicon_matcher->Find(node->id)) {
				node->lexicon_state = lexicon_matcher->Value().nextstate;
				node->is_lex_path = true;
			} else {
				node->is_lex_path = false;
				node->lm_lex_score += this->lex_penalty;
			}
		}
	}

	/*
	NOTE: Hotword scores and beta word penalty were accumulated in seperate variable
		  because, these values will not be passed hereditarily to the child nodes.

		  But, the language model and lexicon scores will be passed to the child nodes.
	*/

	if (hotwords_fst && (node->parent->is_hotpath || node->is_start_of_word)) {

		fst::StdVectorFst::StateId state = node->is_start_of_word ? node->hotword_state : node->parent->hotword_state;
		hotwords_matcher->SetState(state);

		if (hotwords_matcher->Find(node->id)) {
			const fst::StdArc& arc = hotwords_matcher->Value();
			/*
			NOTE: Here,
				arc.olabel is the token length so far in the hotword,
				arc.weight.Value() is the weight for each hotword token.
			*/
			node->hotword_state = arc.nextstate;
			node->hw_score = (arc.olabel * arc.weight.Value());
			node->is_hotpath = true;
		}
	}
}

#endif // _ZCTC_EXT_SCORER_H
