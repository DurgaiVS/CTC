V3:
- when repetition handling, if the prob is less, we are not updating the node's prob,
    but this way, lesser prob class is overtaking the squashed prob, and misleading
    the output seq.
- when repetition handling, if the prob is greater, we are updating the timestep and
    changing the prob to (at t value), and also we are adding child to it, which might
    be misleading, like, at t=5 node, we found repetition and updated "t" to 6, also
    another class is getting added to this node
- add to consider only the timestep with max confidence

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
