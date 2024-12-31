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
    std::unordered_map<std::string, int> char_map;

    ZFST(char* vocab_path, char* fst_path)
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

        this->load_vocab(vocab_path);
    }

    explicit ZFST(char* vocab_path, fst::StdVectorFst* fst)
        : fst(fst)
    {
        this->load_vocab(vocab_path);
    }

    ~ZFST() { delete fst; }

    void insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<std::vector<int>>& tokens_group);
    void optimize();
    int parse_lexicon_files(std::vector<std::string>& file_paths, int freq_threshold, int worker_count);
    int parse_lexicon_file(std::string file_path, int freq_threshold);
    bool write(std::string output_path);

    inline void insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<int>& tokens);

protected:
    inline void load_vocab(char* vocab_path);
};

int
parse_lexicon_file(ZFST* zfst, std::string file_path, int freq_threshold);
// NOTE: hotwords_weight should be sorted in descending order...
void
populate_hotword_fst(fst::StdVectorFst* fst, std::vector<std::vector<int>>& hotwords,
                     std::vector<float>& hotwords_weight);

} // namespace zctc

/* ---------------------------------------------------------------------------- */

int
zctc::ZFST::parse_lexicon_file(std::string file_path, int freq_threshold)
{
    return zctc::parse_lexicon_file(this, file_path, freq_threshold);
}

int
zctc::ZFST::parse_lexicon_files(std::vector<std::string>& file_paths, int freq_threshold, int worker_count)
{
    ThreadPool pool(worker_count);
    std::vector<std::future<int>> results;

    for (std::string file_path : file_paths)
        results.emplace_back(pool.enqueue(zctc::parse_lexicon_file, this, file_path, freq_threshold));

    for (auto&& result : results)
        if (result.get() != 0)
            throw std::runtime_error("Unexpected error occured during execution");

    return 0;
}

bool
zctc::ZFST::write(std::string output_path)
{
    std::lock_guard<std::mutex> guard(this->mutex);

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

void
zctc::ZFST::insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<int>& tokens)
{
    fst::StdVectorFst::StateId next_state, state = this->fst->Start();

    for (int token : tokens) {
        std::lock_guard<std::mutex> guard(this->mutex);

        matcher->SetState(state);
        if (matcher->Find(token)) {
            state = matcher->Value().nextstate;
            continue;

        } else {
            next_state = this->fst->AddState();
            this->fst->AddArc(state, fst::StdArc(token, token, 0, next_state));
            state = next_state;
        }
    }

    std::lock_guard<std::mutex> guard(this->mutex);
    this->fst->SetFinal(state, 0);
}

void
zctc::ZFST::insert_into_fst(fst::SortedMatcher<fst::StdVectorFst>* matcher, std::vector<std::vector<int>>& tokens_group)
{
    fst::StdVectorFst::StateId next_state, state;
    for (std::vector<int>& tokens : tokens_group) {
        state = this->fst->Start();

        for (int token : tokens) {
            std::lock_guard<std::mutex> guard(this->mutex);

            matcher->SetState(state);
            if (matcher->Find(token)) {
                state = matcher->Value().nextstate;
                continue;

            } else {
                next_state = this->fst->AddState();
                this->fst->AddArc(state, fst::StdArc(token, token, 0, next_state));
                state = next_state;
            }
        }

        std::lock_guard<std::mutex> guard(this->mutex);
        this->fst->SetFinal(state, 0);
    }
}

int
zctc::parse_lexicon_file(zctc::ZFST* zfst, std::string file_path, int freq_threshold)
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
            tokens.push_back(zfst->char_map[tmp]);
        }

        zfst->insert_into_fst(&matcher, tokens);
        tokens.clear();
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

void
zctc::ZFST::load_vocab(char* vocab_path)
{

    int id = 0;
    std::string line;
    std::ifstream inputFile(vocab_path);

    if (!inputFile.is_open())
        std::runtime_error("Cannot open vocab file from the path provided.");

    while (std::getline(inputFile, line))
        this->char_map[line] = id++;
}

#endif // _ZCTC_ZFST_H
