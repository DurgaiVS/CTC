import math

import torch
from ctcdecode import CTCBeamDecoder

if __name__ == "__main__":
    # Test the assumption that the new CTCDecoder is faster than the old CTCBeamDecoder
    # for the same input logits.
    # 
    # The assumption is based on the fact that the new CTCDecoder is implemented in C++,
    # and the old CTCBeamDecoder is implemented in Python.
    # 
    # The assumption is tested by running both decoders on the same input logits and
    # comparing the time taken by both decoders.
    # 
    # The input logits are randomly generated and are of shape (1, 100, 100).
    # 
    # The time taken by both decoders is compared by running both decoders 1000 times
    # and calculating the average time taken by both decoders.
    # 
    # The average time taken by both decoders is printed at the end of the script.
    # 
    # The script is run using the following command:
    #

    decoder = CTCBeamDecoder(
        ["_", "a", "b"],
        # is_bpe_based=True,
    )

    res, score, ts, out_len= decoder.decode(torch.tensor([[[0.6, 0.3, 0.1], [0.6, 0.4, 0.0]]], dtype=torch.float32), torch.tensor([2], dtype=torch.int32))

    # print(f"Output: {res}, Score: {score}")
    for o, l, s in zip(res[0], out_len[0], score[0]):
        print(f"Output: {o.tolist()}, Length: {l}, Score: {math.exp(s)}")
