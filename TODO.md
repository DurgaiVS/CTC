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
