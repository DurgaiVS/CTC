from time import time
import torch
from zctc import Decoder
from ctcdecode._ext import ctc_decode

def new_ctc(batch_size, seq_len, vocab_size, beam_width, cutoff_top_n, cutoff_prob, blank_id, vocab, logits, sorted_indices): #, labels, timesteps, scores, out_seq_len):
    
    labels = torch.zeros(batch_size, beam_width, seq_len, device="cpu").int().numpy()
    timesteps = torch.zeros(batch_size, beam_width, seq_len, device="cpu").int().numpy()
    # scores = torch.FloatTensor(batch_size, beam_width, device="cpu").float()
    # out_seq_len = torch.zeros(batch_size, beam_width, device="cpu").int()

    decoder = Decoder(batch_size, blank_id, cutoff_top_n, vocab_size, cutoff_prob, beam_width)
    decoder.batch_decode(logits.numpy(), sorted_indices.numpy(), labels, timesteps, logits.shape[0], logits.shape[1])

    return torch.from_numpy(labels), torch.from_numpy(timesteps)

def old_ctc(batch_size, seq_len, vocab_size, beam_width, cutoff_top_n, cutoff_prob, blank_id, vocab, logits, sorted_indices): #, labels, timesteps, scores, out_seq_len):

    labels = torch.zeros(batch_size, beam_width, seq_len, device="cpu").int()
    timesteps = torch.IntTensor(batch_size, beam_width, seq_len, device="cpu").int()
    scores = torch.FloatTensor(batch_size, beam_width, device="cpu").float()
    out_seq_len = torch.zeros(batch_size, beam_width, device="cpu").int()

    decoder = ctc_decode.paddle_get_decoder(vocab, cutoff_top_n, cutoff_prob, beam_width, batch_size, blank_id, 
        False,
        False,
        -5,
        "#",
        "CTCBeamDecoderLogger.txt",
        "none"
    )
    decoder_input = ctc_decode.get_decoder_input(logits, torch.IntTensor([seq_len]))
    ctc_decode.paddle_beam_decode(decoder_input, decoder, labels, timesteps, scores, out_seq_len)

    return labels, timesteps

if __name__ == "__main__":

    batch_size = 1
    seq_len = 12
    vocab_size = 8
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h"]
    beam_width = 1
    cutoff_top_n = 20
    cutoff_prob = 0.95
    blank_id = 0

    logits = torch.randn((batch_size, seq_len, vocab_size), device="cpu", dtype=torch.float32).uniform_()
    sorted_indices = torch.argsort(logits, dim=2, descending=True).to(torch.int32)

    start = time()
    op_old, ts_old = old_ctc(batch_size, seq_len, vocab_size, beam_width, cutoff_top_n, cutoff_prob, blank_id, vocab, logits, sorted_indices) #, labels, timesteps, scores, out_seq_len)
    end = time()

    print("Old-CTC:", end - start, "seconds")

    start = time()
    op_new, ts_new = new_ctc(batch_size, seq_len, vocab_size, beam_width, cutoff_top_n, cutoff_prob, blank_id, vocab, logits, sorted_indices) #, labels, timesteps, scores, out_seq_len)
    end = time()

    print("New-CTC:", end - start, "seconds")
    try:
        print(torch.all(op_old[op_old > 0.5] == op_new[op_new > 0.5]))
    except Exception:
        print(logits, "\n", sorted_indices, "\n", op_old[0])
        # print(op_new[1], "\n", op_old[1][0], "\n", torch.argmax(logits, dim=2)[1], "\n", sorted_indices[1][:,0], end="\nERROR\n")
    else:
        print(op_new[0], "\n", op_old[0][0], "\n", torch.argmax(logits, dim=2)[0], "\n", sorted_indices[0][:,0], end="\nEND\n")
