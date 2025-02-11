#ifndef _ZCTC_EXT_SCORER_H
#define _ZCTC_EXT_SCORER_H

#include "fst/fstlib.h"
#include "lm/model.hh"

#include "./node.hh"

namespace zctc {

class ExternalScorer {
public:
	const char tok_sep;
	const int apostrophe_id;
	const float alpha, penalty;
	lm::base::Model* lm;
	fst::StdVectorFst* lexicon;

	/*
	NOTE: The `alpha` passed here should be in log(base 10) scale.
	*/
	ExternalScorer(char tok_sep, int apostrophe_id, float alpha, float penalty, char* lm_path, char* lexicon_path)
		: tok_sep(tok_sep)
		, apostrophe_id(apostrophe_id)
		, alpha(alpha)
		, penalty(penalty)
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
	inline void start_of_word_check(Node<T>* node, fst::StdVectorFst* hotwords_fst) const;

	template <typename T>
	inline void initialise_start_states(Node<T>* root, fst::StdVectorFst* hotwords_fst) const;

	template <typename T>
	void run_ext_scoring(zctc::Node<T>* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
						 fst::StdVectorFst* hotwords_fst,
						 fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const;
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
void
zctc::ExternalScorer::start_of_word_check(Node<T>* node, fst::StdVectorFst* hotwords_fst) const
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

template <typename T>
void
zctc::ExternalScorer::initialise_start_states(Node<T>* root, fst::StdVectorFst* hotwords_fst) const
{
	if (this->lexicon)
		root->lexicon_state = this->lexicon->Start();

	if (this->lm)
		this->lm->BeginSentenceWrite(&(root->lm_state));

	if (hotwords_fst)
		root->hotword_state = hotwords_fst->Start();
}

template <typename T>
void
zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* node, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
									  fst::StdVectorFst* hotwords_fst,
									  fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const
{

	if (this->lm) {

		lm::WordIndex word_id = this->lm->BaseVocabulary().Index(node->token);

		/*
		NOTE: If I just give penalty for OOV, the state will not
			  be updated. Also, since the `lm_prob` is in log scale,
			  the penalty could be bigger than the `lm_prob` itself.
			  Assuming a penalty of -5.0, the `lm_prob` could be -10.0,
			  which is a huge difference. So just take the `lm_prob`
			  as it is.
		*/

		if (word_id == this->lm->BaseVocabulary().NotFound()) { // OOV char
			node->lm_prob = -100;
		} else {
			/*
			NOTE: Since KenLM returns the log probability with base 10,
				  convert the log probability to linear scale.
			*/
			node->lm_prob = std::pow(
				10.0f, this->alpha + this->lm->BaseScore(&(node->parent->lm_state), word_id, &(node->lm_state)));
		}
	}

	this->start_of_word_check(node, hotwords_fst);

	if (this->lexicon) {

		if (!(node->parent->is_lex_path || node->is_start_of_word)) {

			node->is_lex_path = false;

		} else {

			fst::StdVectorFst::StateId state
				= node->is_start_of_word ? node->lexicon_state : node->parent->lexicon_state;
			lexicon_matcher->SetState(state);

			if (lexicon_matcher->Find(node->id)) {
				node->lexicon_state = lexicon_matcher->Value().nextstate;
				node->is_lex_path = true;
			} else {
				node->is_lex_path = false;
			}
		}
	}

	if (hotwords_fst && (node->parent->is_hotpath || node->is_start_of_word)) {

		fst::StdVectorFst::StateId state = node->is_start_of_word ? node->hotword_state : node->parent->hotword_state;
		hotwords_matcher->SetState(state);

		if (hotwords_matcher->Find(node->id)) {
			const fst::StdArc& arc = hotwords_matcher->Value();
			node->hotword_length = arc.olabel;
			node->hotword_weight = arc.weight.Value();
			node->is_hotpath = true;
		}
	}
}

#endif // _ZCTC_EXT_SCORER_H
