#include "zctc/decoder.hh"

PYBIND11_MODULE(_zctc, m) {
    py::class_<Decoder>(m, "Decoder")
       .def(py::init<int, int, int, int, float, py::ssize_t>())
       .def("batch_decode", &Decoder::batch_decode<float>,
            py::call_guard<py::gil_scoped_release>())
       .def("batch_decode", &Decoder::batch_decode<double>,
            py::call_guard<py::gil_scoped_release>())
       .def_readonly("blank_id", &Decoder::blank_id)
       .def_readonly("beam_width", &Decoder::beam_width)
       .def_readonly("cutoff_top_n", &Decoder::cutoff_top_n)
       .def_readonly("thread_count", &Decoder::thread_count)
       .def_readonly("nucleus_prob_per_timestep", &Decoder::nucleus_prob_per_timestep);
}
