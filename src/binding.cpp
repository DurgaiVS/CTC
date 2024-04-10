#include "zctc/decoder.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(_zctc, m) {
    pybind11::class_<Decoder>(m, "Decoder", pybind11::call_guard<pybind11::gil_scoped_release>())
       .def(pybind11::init<int, int, int, int, float>())
       .def("batch_decode", &Decoder::batch_decode<float>,
            pybind11::call_guard<pybind11::gil_scoped_release>())
       .def("batch_decode", &Decoder::batch_decode<double>,
            pybind11::call_guard<pybind11::gil_scoped_release>())
       .def_readonly("blank_id", &Decoder::blank_id)
       .def_readonly("beam_width", &Decoder::beam_width)
       .def_readonly("cutoff_top_n", &Decoder::cutoff_top_n)
       .def_readonly("thread_count", &Decoder::thread_count)
       .def_readonly("nucleus_prob_per_timestep", &Decoder::nucleus_prob_per_timestep);
}
