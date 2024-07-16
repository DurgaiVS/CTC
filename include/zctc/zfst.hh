#ifndef _ZCTC_ZFST_H
#define _ZCTC_ZFST_H

#include <filesystem>
#include <mutex>
#include <vector>

#include <ThreadPool.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fst/fst.h"
#include "fst/fstlib.h"

namespace zctc {

void
init_fst(fst::StdVectorFst* fst);

class ZFST {
public:
    fst::StdVectorFst* fst;
    std::mutex mutex;

    ZFST(char* fst_path)
        : fst(nullptr)
    {
        if (fst_path) {
            this->fst = fst::StdVectorFst::Read(fst_path);
            if (!this->fst)
                throw std::runtime_error(std::string("Failed to read FST file from the path, ") + fst_path);
        } else {
            this->fst = new fst::StdVectorFst;
            init_fst(this->fst);
        }
    }

    explicit ZFST(fst::StdVectorFst* fst)
        : fst(fst)
    {
    }

    ~ZFST() { delete fst; }

    void optimize();
    int parse_lexicon_file(std::string file_path, int freq_threshold, std::unordered_map<std::string, int>& char_map,
                           int worker_count);
    int parse_lexicon_files(std::vector<std::string>& file_paths, int freq_threshold,
                            std::unordered_map<std::string, int>& char_map, int worker_count);
    void insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<int>& tokens);
    void insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<std::vector<int>>& tokens_group);
    bool write(std::string output_path);

protected:
    fst::StdVectorFst::StateId insert_path(fst::StdVectorFst::StateId state, int value);
    fst::StdVectorFst::StateId is_path_available(fst::SortedMatcher<fst::StdVectorFst>* matcher,
                                                 fst::StdVectorFst::StateId state, int value);
};

int
parse_lexicon_file(ZFST* zfst, std::string file_path, int freq_threshold,
                   std::unordered_map<std::string, int>& char_map);
// NOTE: hotwords_weight should be sorted in descending order...
void
populate_hotword_fst(fst::StdVectorFst* fst, std::vector<std::vector<int>>& hotwords,
                     std::vector<float>& hotwords_weight);

} // namespace zctc

/* ---------------------------------------------------------------------------- */

int
zctc::ZFST::parse_lexicon_file(std::string file_path, int freq_threshold,
                               std::unordered_map<std::string, int>& char_map, int worker_count)
{
    return zctc::parse_lexicon_file(this, file_path, freq_threshold, char_map);
}

int
zctc::ZFST::parse_lexicon_files(std::vector<std::string>& file_paths, int freq_threshold,
                                std::unordered_map<std::string, int>& char_map, int worker_count)
{
    ThreadPool pool(worker_count);
    std::vector<std::future<int>> results;

    for (std::string file_path : file_paths)
        results.emplace_back(pool.enqueue(zctc::parse_lexicon_file, this, file_path, freq_threshold, char_map));

    for (auto&& result : results)
        if (result.get() != 0)
            throw std::runtime_error("Unexpected error occured during execution");

    return 0;
}

bool
zctc::ZFST::write(std::string output_path)
{
    return this->fst->Write(output_path);
}

void
zctc::init_fst(fst::StdVectorFst* fst)
{
    if (fst->NumStates() != 0)
        return;

    fst::StdVectorFst::StateId start_state = fst->AddState();
    assert(start_state == 0);
    fst->SetStart(start_state);
}

void
zctc::ZFST::optimize()
{
    std::lock_guard<std::mutex> guard(this->mutex);

    fst::RmEpsilon(this->fst);
    fst::Determinize(*(this->fst), this->fst);
    fst::Minimize(this->fst);
}

fst::StdVectorFst::StateId
zctc::ZFST::is_path_available(fst::SortedMatcher<fst::StdVectorFst>* matcher, fst::StdVectorFst::StateId state,
                              int value)
{
    std::lock_guard<std::mutex> guard(this->mutex);

    matcher->SetState(state);
    if (matcher->Find(value))
        return matcher->Value().nextstate;
    else
        return this->fst->Start();
}

fst::StdVectorFst::StateId
zctc::ZFST::insert_path(fst::StdVectorFst::StateId state, int value)
{
    std::lock_guard<std::mutex> guard(this->mutex);

    fst::StdVectorFst::StateId next_state = this->fst->AddState();
    this->fst->AddArc(state, fst::StdArc(value, value, 0, next_state));
    return next_state;
}

void
zctc::ZFST::insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<int>& tokens)
{
    fst::StdVectorFst::StateId next_state, state = this->fst->Start();

    for (int token : tokens) {
        next_state = this->is_path_available(matcher, state, token);
        if (next_state == this->fst->Start())
            state = this->insert_path(state, token);
        else
            state = next_state;
    }
}

void
zctc::ZFST::insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<std::vector<int>>& tokens_group)
{
    fst::StdVectorFst::StateId next_state, state;
    for (std::vector<int>& tokens : tokens_group) {
        state = this->fst->Start();

        for (int token : tokens) {
            next_state = this->is_path_available(matcher, state, token);
            if (next_state == this->fst->Start())
                state = this->insert_path(state, token);
            else
                state = next_state;
        }
    }
}

int
zctc::parse_lexicon_file(zctc::ZFST* zfst, std::string file_path, int freq_threshold,
                         std::unordered_map<std::string, int>& char_map)
{
    int freq;
    std::string word, tmp, line;
    std::vector<int> tokens;
    fst::SortedMatcher<fst::StdVectorFst> matcher(zfst->fst, fst::MATCH_INPUT);

    std::ifstream file(file_path);
    if (!file) {
        std::runtime_error(std::string("Failed to read FST file from the path, ") + file_path);
    }

    while (std::getline(file, line)) {
        // Lexicon file format:
        // freq-count actual-word *tokenized-version-of-the-word
        // 1 the t ##h ##e

        std::istringstream iss(line);
        iss >> freq;
        iss >> word;
        if (freq < freq_threshold) {
            continue;
        }

        while (iss.good()) {
            iss >> tmp;
            tokens.push_back(char_map[tmp]);
        }

        zfst->insert_into_fst(&matcher, tokens);
    }

    return 0;
}

void
zctc::populate_hotword_fst(fst::StdVectorFst* fst, std::vector<std::vector<int>>& hotwords,
                           std::vector<float>& hotwords_weight)
{
    int token;
    float hotword_weight, hotword_split;
    fst::StdVectorFst::StateId state, next_state;
    fst::SortedMatcher<fst::StdVectorFst> matcher(fst, fst::MATCH_INPUT);

    zctc::init_fst(fst);

    for (int i = 0; i < hotwords.size(); i++) {
        std::vector<int>& tokens = hotwords[i];
        hotword_weight = hotwords_weight[i];
        hotword_split = hotword_weight / tokens.size();
        state = fst->Start();

        for (int j = 0; j < tokens.size(); j++) {
            token = tokens[j];
            matcher.SetState(state);

            if (matcher.Find(token)) {
                state = matcher.Value().nextstate;
                continue;
            } else {
                next_state = fst->AddState();
                fst->AddArc(state, fst::StdArc(token, (j + 1), hotword_split, next_state));
                state = next_state;
                continue;
            }
        }
    }
}

#endif // _ZCTC_ZFST_H
