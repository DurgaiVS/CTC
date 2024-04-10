import torch
from zctc import Decoder

logits = torch.randn((1, 12, 8), device="cpu").uniform_()
sorted_indices = torch.argsort(logits, dim=2, descending=True)
labels = torch.zeros(1, 25, 12, device="cpu", dtype=torch.int32).tolist() # batch_size, beam_width, 1
timesteps = torch.zeros(1, 25, 12, device="cpu", dtype=torch.int32).tolist() # batch_size, beam_width, 1

decoder = Decoder(0, 25, 20, 6, 0.9)

decoder.batch_decode(logits.tolist(), sorted_indices.tolist(), labels, timesteps, logits.shape[0], logits.shape[1])

print(labels[0][0])
