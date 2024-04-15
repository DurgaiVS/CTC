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
    lm::base::Model* lm;
    fst::StdVectorFst* lexicon;

    ExternalScorer(bool skip, lm::base::Model* lm, fst::StdVectorFst* lexicon)
    : skip(skip),
      lm(lm),
      lexicon(lexicon)
    { }

    template <typename T>
    void run_ext_scoring(zctc::Node<T>* prefix) const;

};

} // namespace zctc


/* ---------------------------------------------------------------------------- */

zctc::ExternalScorer zctc::ExternalScorer::construct_class(char* lm_path, char* lexicon_path) {
    bool skip = false;
    lm::base::Model* lm = nullptr;
    fst::StdVectorFst* lexicon = nullptr;

    if (lm_path)
        lm = lm::ngram::LoadVirtual(lm_path);


    if (lexicon_path)
        lexicon = fst::StdVectorFst::Read(lexicon_path);

    if (lm == nullptr && lexicon == nullptr)
        skip = true;

    return zctc::ExternalScorer(skip, lm, lexicon);

}


template <typename T>
void zctc::ExternalScorer::run_ext_scoring(zctc::Node<T>* prefix) const {
    if (this->skip) return;

    if (this->lm) {

        lm::WordIndex word_id = this->lm->BaseVocabulary().Index(prefix->token);
        prefix->lm_prob = this->lm->BaseScore(&(prefix->parent->lm_state), word_id, &(prefix->lm_state));

    }

    if (this->lexicon) {

        for (fst::ArcIterator<fst::StdVectorFst> aiter(*(this->lexicon), prefix->parent->lexicon_state); !aiter.Done(); aiter.Next()) {

            const fst::StdArc& arc = aiter.Value();
            if (arc.ilabel == prefix->id) {

                prefix->arc_exist = true;
                prefix->lexicon_state = arc.nextstate;
                break;

            }
        }

    }

    prefix->update_score();

}


#endif // _ZCTC_EXT_SCORER_H
