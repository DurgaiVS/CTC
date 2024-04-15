#include "zctc/decoder.hh"
#include <vector>
#include <iostream>

/*
logits = torch.randn((1, 24, 512), device=0)
sorted_indices = torch.argsort(logits, dim=2, descending=True)
labels = torch.IntTensor(1, 283, 12)
timesteps = torch.IntTensor(1, 283, 12)

decoder = _zctc.Decoder(0, 283, 40, 0.9)

decoder.batch_decode(logits.tolist(), sorted_indices.tolist(), labels.tolist(), timesteps.tolist())

print(labels{0}{0})

tensor([0.5353, 0.9165, 0.8097, 0.6203, 0.0874, 0.9848, 0.0446, 0.8812, 0.8137,
        0.2663, 0.2433, 0.4224, 0.1848, 0.2175, 0.4620, 0.6135, 0.6749, 0.6354,
        0.1522, 0.8025, 0.7270, 0.5360, 0.4133, 0.5677, 0.7610, 0.1949, 0.7802,
        0.5341, 0.2236, 0.1395, 0.4069, 0.0177, 0.6533, 0.3861, 0.8282, 0.5513,
        0.9525, 0.9663, 0.4866, 0.7683, 0.4212, 0.1710, 0.1602, 0.3233, 0.2178,
        0.0122, 0.4226, 0.1858, 0.6859, 0.6689, 0.7617, 0.9743, 0.7460, 0.1548,
        0.0350, 0.9986, 0.4946, 0.3633, 0.6819, 0.6312, 0.6001, 0.4001, 0.9493,
        0.2530, 0.6151, 0.5717, 0.0821, 0.9676, 0.2049, 0.1027, 0.1230, 0.7734,
        0.7413, 0.7183, 0.0020, 0.5462, 0.6670, 0.4526, 0.7845, 0.2931, 0.0568,
        0.4773, 0.0674, 0.6782, 0.4566, 0.4192, 0.1512, 0.8155, 0.2685, 0.9500,
        0.0861, 0.4521, 0.0931, 0.2556, 0.6081, 0.1467]) 
 tensor([5, 1, 7, 2, 3, 0, 4, 6, 0, 7, 6, 3, 1, 2, 5, 4, 3, 4, 0, 1, 7, 5, 6, 2,
        2, 0, 3, 6, 4, 1, 5, 7, 5, 4, 2, 7, 0, 3, 6, 1, 6, 0, 3, 4, 7, 1, 2, 5,
        7, 3, 2, 4, 0, 1, 5, 6, 6, 2, 3, 4, 0, 5, 1, 7, 3, 7, 0, 1, 4, 6, 5, 2,
        6, 0, 1, 4, 3, 5, 7, 2, 7, 3, 1, 4, 5, 6, 2, 0, 1, 6, 3, 0, 5, 7, 4, 2],
       dtype=torch.int32) 
 tensor([[5, 3, 5, 6, 7, 6, 3, 6, 7, 1, 0, 0]], dtype=torch.int32) 
 tensor([[5, 0, 3, 2, 5, 6, 7, 6, 3, 6, 7, 1]], dtype=torch.int32)

*/

int main(int argc, char** argv) {
    std::vector<float> logits(512 * 2, 1);

    std::vector<int> sorted_indices(512 * 2, 1);

    std::vector<int> op = {5, 3, 5, 6, 7, 6, 3, 6, 7, 1, 0, 0};

    std::vector<int> labels(12, 0);
    std::vector<int> timesteps(12, 0);

    zctc::Decoder decoder(1, 0, 15, 0.9, 10, -5.0, '#', "/home/durga-17532/zspeech/inference/clients/python/zspeech/inference/resources/models/lm/kenlm/model.bin", "/home/durga-17532/zspeech/inference/clients/python/zspeech/inference/resources/models/spell_checker/en/lexicon.fst.opt", "/home/durga-17532/zspeech/datasets/vocab/asr_training/train_wpe_512.txt");
    decoder.decode(logits.data(), sorted_indices.data(), labels.data(), timesteps.data(), 2);

    for (int i = 0; i < 12; i++)
        std::cout << labels[i] << " == " << op[i] << std::endl;

    return 0;
}
