import torch
from zctc import Decoder

logits = torch.randn((12, 8), device="cpu").uniform_()
sorted_indices = torch.argsort(logits, dim=1, descending=True).to(torch.int32)
labels = torch.zeros(25, 12, device="cpu", dtype=torch.int32).numpy() # batch_size, beam_width, 1
timesteps = torch.zeros(25, 12, device="cpu", dtype=torch.int32).numpy() # batch_size, beam_width, 1

decoder = Decoder(1, 0, 20, 8, 0.9, 25)

decoder.decode(logits.numpy(), sorted_indices.numpy(), labels, timesteps, logits.shape[0])

print(labels, len(labels), type(labels))
