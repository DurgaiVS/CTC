#ifndef _ZCTC_EXT_SCORER_H
#define _ZCTC_EXT_SCORER_H

#include "fst/fstlib.h"
#include "lm/model.hh"

#include "./trie.hh"

namespace zctc {

class ExternalScorer {
public:
    const char tok_sep;
    const int apostrophe_id;
    const float lm_alpha, penalty;
    lm::base::Model* lm;
    fst::StdVectorFst* lexicon;

    ExternalScorer(char tok_sep, int apostrophe_id, float lm_alpha, float penalty, char* lm_path, char* lexicon_path)
        : tok_sep(tok_sep)
        , apostrophe_id(apostrophe_id)
        , lm_alpha(lm_alpha)
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
    inline void start_of_word_check(Node<T>* prefix, fst::StdVectorFst* hotwords_fst) const;

    template <typename T>
    inline void initialise_start_states(Node<T>* root, fst::StdVectorFst* hotwords_fst) const;

    template <typename T>
    void run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
                         fst::StdVectorFst* hotwords_fst,
                         fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const;
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
void
zctc::ExternalScorer::start_of_word_check(Node<T>* prefix, fst::StdVectorFst* hotwords_fst) const
{
    prefix->is_start_of_word = !(prefix->id == this->apostrophe_id || prefix->parent->id == this->apostrophe_id
                                 || prefix->token.at(0) == this->tok_sep);

    if (!prefix->is_start_of_word)
        return;

    if (this->lexicon)
        prefix->lexicon_state = this->lexicon->Start();

    if (hotwords_fst)
        prefix->hotword_state = hotwords_fst->Start();
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
zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* lexicon_matcher,
                                      fst::StdVectorFst* hotwords_fst,
                                      fst::SortedMatcher<fst::StdVectorFst>* hotwords_matcher) const
{

    if (this->lm) {

        lm::WordIndex word_id = this->lm->BaseVocabulary().Index(prefix->token);

        if (word_id == 0) { // OOV char
            prefix->lm_prob = this->penalty;
        } else {
            prefix->lm_prob
                = this->lm_alpha * this->lm->BaseScore(&(prefix->parent->lm_state), word_id, &(prefix->lm_state));
        }
    }

    this->start_of_word_check(prefix, hotwords_fst);

    if (this->lexicon) {

        if (!(prefix->parent->arc_exist || prefix->is_start_of_word)) {

            prefix->arc_exist = false;

        } else {

            fst::StdVectorFst::StateId state
                = prefix->is_start_of_word ? prefix->lexicon_state : prefix->parent->lexicon_state;
            lexicon_matcher->SetState(state);

            if (lexicon_matcher->Find(prefix->id)) {
                prefix->lexicon_state = lexicon_matcher->Value().nextstate;
                prefix->arc_exist = true;
            }
        }

    } else {
        prefix->arc_exist = true;
    }

    if (hotwords_fst && (prefix->parent->is_hotpath || prefix->is_start_of_word)) {

        fst::StdVectorFst::StateId state
            = prefix->is_start_of_word ? prefix->hotword_state : prefix->parent->hotword_state;
        hotwords_matcher->SetState(state);

        if (hotwords_matcher->Find(prefix->id)) {
            const fst::StdArc& arc = hotwords_matcher->Value();
            prefix->hotword_length = arc.olabel;
            prefix->hotword_weight = arc.weight.Value();
            prefix->is_hotpath = true;
        }
    }
}

#endif // _ZCTC_EXT_SCORER_H
