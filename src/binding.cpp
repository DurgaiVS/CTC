#include "zctc/decoder.hh"


PYBIND11_MODULE(_zctc, m) {
     py::class_<zctc::Decoder>(m, "Decoder")
       .def(py::init<int, int, int, float, py::ssize_t , float, char, char*, char*, char*>())
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
