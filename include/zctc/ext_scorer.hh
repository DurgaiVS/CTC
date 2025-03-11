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
	const float alpha, beta, lex_penalty;
	lm::base::Model* lm;
	fst::StdVectorFst* lexicon;

	/*
	NOTE: The `alpha` passed here should be in log(base 10) scale.
	*/
	ExternalScorer(char tok_sep, int apostrophe_id, float alpha, float beta, float lex_penalty, char* lm_path,
				   char* lexicon_path)
		: tok_sep(tok_sep)
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
	T lm_prob = 0.0, hw_score = 0.0, lex_score = 0.0;

	if (this->lm) {

		lm::WordIndex word_id = this->lm->BaseVocabulary().Index(node->token);

		if (word_id == this->lm->BaseVocabulary().NotFound()) {
			lm_prob = -1000; // OOV char
		} else {
			/*
			NOTE: Since KenLM returns the log probability with base 10,
				converting the log probability to base e, using,

				logb(x) = loga(x) / loga(b)
			*/
			lm_prob = this->alpha
					  * (this->lm->BaseScore(&(node->parent->lm_state), word_id, &(node->lm_state)) / zctc::LOG_A_OF_B);
		}
	}

	this->start_of_word_check(node, hotwords_fst);

	if (this->lexicon) {

		if (!(node->parent->is_lex_path || node->is_start_of_word)) {

			node->is_lex_path = false;
			lex_score = this->lex_penalty;

		} else {

			fst::StdVectorFst::StateId state
				= node->is_start_of_word ? node->lexicon_state : node->parent->lexicon_state;
			lexicon_matcher->SetState(state);

			if (lexicon_matcher->Find(node->id)) {
				node->lexicon_state = lexicon_matcher->Value().nextstate;
				node->is_lex_path = true;
			} else {
				node->is_lex_path = false;
				lex_score = this->lex_penalty;
			}
		}
	}

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
			hw_score = arc.olabel * arc.weight.Value();
			node->is_hotpath = true;
		}
	}

	node->ext_score = lm_prob + lex_score + hw_score + (this->beta * node->seq_length);
}

#endif // _ZCTC_EXT_SCORER_H
