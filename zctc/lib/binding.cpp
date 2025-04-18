#include "zctc/decoder.hh"

PYBIND11_MODULE(_zctc, m)
{
	py::class_<zctc::ExternalScorer>(m, "_ExternalScorer")
		.def(py::init<char, int, double, double, double, char*, char*>(), py::arg("tok_sep"), py::arg("apostrophe_id"),
			 py::arg("alpha"), py::arg("beta"), py::arg("lex_penalty"), py::arg("lm_path") = nullptr,
			 py::arg("lexicon_path") = nullptr)
		.def_readonly("tok_sep", &zctc::ExternalScorer::tok_sep)
		.def_readonly("apostrophe_id", &zctc::ExternalScorer::apostrophe_id)
		.def_readonly("alpha", &zctc::ExternalScorer::alpha)
		.def_readonly("beta", &zctc::ExternalScorer::beta)
		.def_readonly("lex_penalty", &zctc::ExternalScorer::lex_penalty);

	py::class_<fst::StdVectorFst>(m, "_Fst")
		.def(pybind11::init<>())
		.def("NumStates", &fst::StdVectorFst::NumStates, "Gets the number of states in the FST")
		.def("Start", &fst::StdVectorFst::Start, "Gets the start state of the FST")
		.def("Final", &fst::StdVectorFst::Final, "Gets the final state of the FST");

	py::class_<zctc::Decoder>(m, "_Decoder")
		.def(py::init<int, int, int, int, double, double, double, py::ssize_t, double, double, double, char,
					  std::vector<std::string>, char*, char*>(),
			 py::arg("thread_count"), py::arg("blank_id"), py::arg("cutoff_top_n"), py::arg("apostrophe_id"),
			 py::arg("nucleus_prob_per_timestep"), py::arg("alpha"), py::arg("beta"), py::arg("beam_width"),
			 py::arg("lex_penalty"), py::arg("min_tok_prob"), py::arg("max_beam_score_deviation"), py::arg("tok_sep"),
			 py::arg("vocab"), py::arg("lm_path") = nullptr, py::arg("lexicon_path") = nullptr)
		.def("generate_hw_fst", &zctc::Decoder::generate_hw_fst, py::arg("hotwords_id"), py::arg("hotwords_weight"),
			 py::arg("hotwords_fst") = nullptr, pybind11::return_value_policy::take_ownership,
			 py::call_guard<py::gil_scoped_release>())
		.def("batch_decode", &zctc::Decoder::batch_decode_wrapper, py::arg("logits"), py::arg("logit_bytes"),
			 py::arg("ids"), py::arg("labels"), py::arg("timesteps"), py::arg("seq_len"), py::arg("seq_pos"),
			 py::arg("batch_size"), py::arg("max_seq_len"), py::arg("hotwords") = std::vector<std::vector<int>>(),
			 py::arg("hotwords_weight") = std::vector<float>(), py::arg("hotwords_fst") = nullptr,
			 py::call_guard<py::gil_scoped_release>())

#ifndef NDEBUG
		// NOTE: This function is only for debugging purpose.
		.def("serial_decode", &zctc::Decoder::serial_decode_wrapper, py::arg("logits"), py::arg("logit_bytes"),
			 py::arg("ids"), py::arg("labels"), py::arg("timesteps"), py::arg("seq_len"), py::arg("seq_pos"),
			 py::arg("batch_size"), py::arg("max_seq_len"), py::arg("hotwords") = std::vector<std::vector<int>>(),
			 py::arg("hotwords_weight") = std::vector<float>(), py::arg("hotwords_fst") = nullptr,
			 py::call_guard<py::gil_scoped_release>())
#endif // NDEBUG

		.def_readonly("blank_id", &zctc::Decoder::blank_id)
		.def_readonly("beam_width", &zctc::Decoder::beam_width)
		.def_readonly("cutoff_top_n", &zctc::Decoder::cutoff_top_n)
		.def_readonly("thread_count", &zctc::Decoder::thread_count)
		.def_readonly("vocab_size", &zctc::Decoder::vocab_size)
		.def_readonly("min_tok_prob", &zctc::Decoder::min_tok_prob)
		.def_readonly("max_beam_score_deviation", &zctc::Decoder::max_beam_score_deviation)
		.def_readonly("nucleus_prob_per_timestep", &zctc::Decoder::nucleus_prob_per_timestep)
		.def_readonly("vocab", &zctc::Decoder::vocab)
		.def_readonly("ext_scorer", &zctc::Decoder::ext_scorer);

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
