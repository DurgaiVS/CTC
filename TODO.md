V2:

- hotwords inference should be similar to language model inference
    at any given point, previous few tokens(till start of the word) will be taken
    and inferred with hotwords, and return prob weights
- pass vectors from python as reference instead of just value,
    refer pybind11's numpy docs(also try to vectorize if possible)

V1:

- Loop through all the nodes at depth D, a method for trie,
- Divide and conquer for decoding a single element from a batch
- Python interface, by exposing directly the decoder class
- LM, lexicon, hotwords support
- For hotword, update the weight asper length from python side,
    and create WFST for the arcs
