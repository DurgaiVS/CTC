# ZCTC

`zctc` is a CTC beam search decoding library written in C++ and integrated into Python using pybind. This library provides efficient and accurate decoding for Connectionist Temporal Classification (CTC) based models.

## Features

- High-performance CTC beam search decoding
- Seamless integration with Python via pybind
- Easy to use API

## Features yet to include

- Currently `softmax` inputs only are supported, have to provide support for `log_softmax` and `unnormalized` inputs too.
- Currently this package only accepts BPE tokenized vocabulary, still have to add Character vocab and any other as per requirements.
- Should include tests to ensure seamless working of the CTC logic.

## Installation

This package is still under development, and this is still a beta version. So, this package is not yet published in `pypi`.

## Usage

Please refer [this](./test.py) file for usage example.

In that file, we've compared the time stats between [parlance](https://github.com/parlance/ctcdecode/tree/master) version of CTCBeamDecoding and ours. With the involvement of both KenLM and Lexicon, we found ours nearly `3` to `4` times faster for longer sequence inputs. We've also made few other changes from their version of decoder, like, we operate on `numpy arrays` instead of `torch tensors` to make the torch dependency loosely bound, etc...

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Acknowledgements

- [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) for the original algorithm

## Contact

For any questions or inquiries, please contact [durgaivelselvan.mn@zohocorp.com].
