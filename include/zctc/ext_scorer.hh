#ifndef _ZCTC_EXT_SCORER_H
#define _ZCTC_EXT_SCORER_H

#include "fst/fstlib.h"
#include "lm/model.hh"
#include "lm/vocab.hh"

#include "./trie.hh"

namespace zctc {

class ExternalScorer {
public:

    static ExternalScorer construct_class(char* lm_path, char* lexicon_path);

    bool skip;
    lm::ngram::QuantArrayTrieModel* lm;
    fst::StdVectorFst* lexicon;

    ExternalScorer(bool skip, lm::ngram::QuantArrayTrieModel* lm, fst::StdVectorFst* lexicon)
    : skip(skip),
      lm(lm),
      lexicon(lexicon)
    { }

    ~ExternalScorer() {
        delete this->lm;
        delete this->lexicon;
    }

    template <typename T>
    void run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* matcher) const;

};

} // namespace zctc


/* ---------------------------------------------------------------------------- */

zctc::ExternalScorer zctc::ExternalScorer::construct_class(char* lm_path, char* lexicon_path) {
    bool skip = false;
    lm::ngram::QuantArrayTrieModel* lm = nullptr;
    fst::StdVectorFst* lexicon = nullptr;

    if (lm_path)
        lm = new lm::ngram::QuantArrayTrieModel(lm_path);


    if (lexicon_path)
        lexicon = fst::StdVectorFst::Read(lexicon_path);

    if (lm == nullptr && lexicon == nullptr)
        skip = true;

    return zctc::ExternalScorer(skip, lm, lexicon);

}


template <typename T>
void zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* prefix, fst::SortedMatcher<fst::StdVectorFst>* matcher) const {
    if (this->skip) return;

    if (this->lm) {

        lm::WordIndex word_id = this->lm->BaseVocabulary().Index(prefix->token);
        prefix->lm_prob = this->lm->BaseScore(&(prefix->parent->lm_state), word_id, &(prefix->lm_state));

    }

    if (this->lexicon) {

        if (!(prefix->parent->arc_exist || prefix->is_start_of_word)) {

            prefix->arc_exist = false;

        } else {

            fst::StdVectorFst::StateId state = prefix->is_start_of_word ? prefix->lexicon_state : prefix->parent->lexicon_state;
            matcher->SetState(state);

            if (matcher->Find(prefix->id)) {
                prefix->lexicon_state = matcher->Value().nextstate;
                prefix->arc_exist = true;
            }

        }

    }

    prefix->update_score();

}


#endif // _ZCTC_EXT_SCORER_H
