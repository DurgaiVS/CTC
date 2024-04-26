#include "zctc/decoder.hh"


PYBIND11_MODULE(_zctc, m) {
     py::class_<zctc::Decoder>(m, "_Decoder")
       .def(py::init<int, int, int, int, float, float, py::ssize_t, float, char, std::vector<std::string>, char*, char*>(),
           py::arg("thread_count"), py::arg("blank_id"), 
           py::arg("cutoff_top_n"), py::arg("apostrophe_id"),
           py::arg("nucleus_prob_per_timestep"), py::arg("lm_alpha"), 
           py::arg("beam_width"), py::arg("penalty"), 
           py::arg("tok_sep"), py::arg("vocab"),
           py::arg("lm_path") = nullptr, py::arg("lexicon_path") = nullptr)
       .def("batch_decode", &zctc::Decoder::batch_decode<float>,
            py::call_guard<py::gil_scoped_release>())
       .def("batch_decode", &zctc::Decoder::batch_decode<double>,
            py::call_guard<py::gil_scoped_release>())
       .def_readonly("blank_id", &zctc::Decoder::blank_id)
       .def_readonly("beam_width", &zctc::Decoder::beam_width)
       .def_readonly("cutoff_top_n", &zctc::Decoder::cutoff_top_n)
       .def_readonly("thread_count", &zctc::Decoder::thread_count)
       .def_readonly("vocab_size", &zctc::Decoder::vocab_size)
       .def_readonly("penalty", &zctc::Decoder::penalty)
       .def_readonly("nucleus_prob_per_timestep", &zctc::Decoder::nucleus_prob_per_timestep);
}
