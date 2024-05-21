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
    const float lm_alpha;
    lm::base::Model* lm;
    fst::StdVectorFst *lexicon, *hw_fst;

    ExternalScorer(char tok_sep, int apostrophe_id, float lm_alpha, char* lm_path, char* lexicon_path)
        : tok_sep(tok_sep)
        , apostrophe_id(apostrophe_id)
        , lm_alpha(lm_alpha)
        , lm(nullptr)
        , lexicon(nullptr)
        , hw_fst(nullptr)
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

        if (this->hw_fst)
            delete this->hw_fst;
    }

    template <typename T>
    inline void start_of_word_check(Node<T>* prefix) const;

    template <typename T>
    inline void initialise_start_states(Node<T>* root) const;

    template <typename T>
    void run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* matcher) const;
};

} // namespace zctc

/* ---------------------------------------------------------------------------- */

template <typename T>
void
zctc::ExternalScorer::start_of_word_check(Node<T>* prefix) const
{
    prefix->is_start_of_word = !(prefix->id == this->apostrophe_id || prefix->parent->id == this->apostrophe_id
                                 || prefix->token.at(0) == this->tok_sep);

    if (prefix->is_start_of_word)
        prefix->lexicon_state = this->lexicon->Start();
}

template <typename T>
void
zctc::ExternalScorer::initialise_start_states(Node<T>* root) const
{
    if (this->lexicon)
        root->lexicon_state = this->lexicon->Start();

    if (this->lm)
        this->lm->NullContextWrite(&(root->lm_state));
}

template <typename T>
void
zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* matcher) const
{

    if (this->lm) {

        lm::WordIndex word_id = this->lm->BaseVocabulary().Index(prefix->token);
        prefix->lm_prob
            = this->lm_alpha + this->lm->BaseScore(&(prefix->parent->lm_state), word_id, &(prefix->lm_state));
    }

    if (this->lexicon) {

        this->start_of_word_check(prefix);

        if (!(prefix->parent->arc_exist || prefix->is_start_of_word)) {

            prefix->arc_exist = false;

        } else {

            fst::StdVectorFst::StateId state
                = prefix->is_start_of_word ? prefix->lexicon_state : prefix->parent->lexicon_state;
            matcher->SetState(state);

            if (matcher->Find(prefix->id)) {
                prefix->lexicon_state = matcher->Value().nextstate;
                prefix->arc_exist = true;
            }
        }

    } else {
        prefix->arc_exist = true;
    }

    if (this->hw_fst) { }
}

#endif // _ZCTC_EXT_SCORER_H
