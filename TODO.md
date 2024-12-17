V4:
    - hotwords, (can have seperate hotwords_weight(T) & is_complete_hotword(bool), concatenate hotword weight)
    - move kenlm libs to sys.path's lib folder, and _zctc within package
    - merge zfst into this
    - On old repo, noted that, the decoding logic without any external scorers is not similar to argmax,
        (excluding monotonic behaviours)...
    - use token id as input symbol and N th token as output symbol, so when checking for arc, we can find where it comes and then give hotword weight, like,
    input symbol - token id
    output symbol - its position in a tokenized word. So when inferencing, store the token position and weight with it...
    - Now have score & score_hw seperately and score_hw = score + (arc.olabel * arc.weight) where arc is the output from the matcher for the next token id.....
    - hotwords weight should be sorted in descending order, and for same hotword weight, it must be sorted on token length in ascending order.
    - include `parse_lexicon_file` on ZFST and add it to binding.cpp, so it can be independently called from python side.


V5:
    - The CTC monotonic logic (accumulating probs paths leading to the same decoded output)
    - If we encounter two different paths leading to the same decoded output at any timepoint, we have to use the least confident paths probs and add it to the most confident path, and have to drop the path itself, (like only considering the probs of it, and the reason for dropping it is due to the viterbi algorithm....) when taking the beamwidth number of max probs paths to next timestep...
