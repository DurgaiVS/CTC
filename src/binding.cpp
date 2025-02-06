#include "zctc/decoder.hh"

PYBIND11_MODULE(_zctc, m)
{
	py::class_<zctc::Decoder>(m, "_Decoder")
		.def(py::init<int, int, int, int, float, float, float, py::ssize_t, float, char, std::vector<std::string>,
					  char*, char*>(),
			 py::arg("thread_count"), py::arg("blank_id"), py::arg("cutoff_top_n"), py::arg("apostrophe_id"),
			 py::arg("nucleus_prob_per_timestep"), py::arg("alpha"), py::arg("beta"), py::arg("beam_width"),
			 py::arg("penalty"), py::arg("tok_sep"), py::arg("vocab"), py::arg("lm_path") = nullptr,
			 py::arg("lexicon_path") = nullptr)
		.def("batch_decode", &zctc::Decoder::batch_decode<float>, py::arg("batch_log_logits"),
			 py::arg("batch_sorted_ids"), py::arg("batch_labels"), py::arg("batch_timesteps"), py::arg("batch_seq_len"),
			 py::arg("batch_seq_pos"), py::arg("batch_size"), py::arg("max_seq_len"),
			 py::arg("hotwords") = std::vector<std::vector<int>>(), py::arg("hotwords_weight") = std::vector<float>(),
			 py::call_guard<py::gil_scoped_release>())
		.def("batch_decode", &zctc::Decoder::batch_decode<double>, py::arg("batch_log_logits"),
			 py::arg("batch_sorted_ids"), py::arg("batch_labels"), py::arg("batch_timesteps"), py::arg("batch_seq_len"),
			 py::arg("batch_seq_pos"), py::arg("batch_size"), py::arg("max_seq_len"),
			 py::arg("hotwords") = std::vector<std::vector<int>>(), py::arg("hotwords_weight") = std::vector<float>(),
			 py::call_guard<py::gil_scoped_release>())
		.def_readonly("blank_id", &zctc::Decoder::blank_id)
		.def_readonly("beta", &zctc::Decoder::beta)
		.def_readonly("beam_width", &zctc::Decoder::beam_width)
		.def_readonly("cutoff_top_n", &zctc::Decoder::cutoff_top_n)
		.def_readonly("thread_count", &zctc::Decoder::thread_count)
		.def_readonly("vocab_size", &zctc::Decoder::vocab_size)
		.def_readonly("penalty", &zctc::Decoder::penalty)
		.def_readonly("nucleus_prob_per_timestep", &zctc::Decoder::nucleus_prob_per_timestep);

	py::class_<zctc::ZFST>(m, "_ZFST")
		.def(py::init<char*, char*>(), py::arg("vocab_path"), py::arg("fst_path") = nullptr)
		//    .def(py::init<fst::StdVectorFst*>(), py::arg("fst"))
		.def("parse_lexicon_files", &zctc::ZFST::parse_lexicon_files, py::arg("file_paths"), py::arg("freq_threshold"),
			 py::arg("worker_count"), py::call_guard<py::gil_scoped_release>())
		.def("parse_lexicon_file", &zctc::ZFST::parse_lexicon_file, py::arg("file_path"), py::arg("freq_threshold"),
			 py::call_guard<py::gil_scoped_release>())
		.def("optimize", &zctc::ZFST::optimize)
		.def("write", &zctc::ZFST::write, py::arg("output_path"))
		.def_readonly("char_map", &zctc::ZFST::char_map);
}
