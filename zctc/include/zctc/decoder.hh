#ifndef _ZCTC_DECODER_H
#define _ZCTC_DECODER_H

#include "ThreadPool.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "./ext_scorer.hh"
#include "./node.hh"
#include "./zfst.hh"

namespace py = pybind11;

namespace zctc {

class Decoder {
public:
	static bool descending_compare(zctc::Node* x, zctc::Node* y);

	const int thread_count, blank_id, cutoff_top_n, vocab_size;
	const double nucleus_prob_per_timestep, min_tok_prob, max_beam_score_deviation;
	const std::size_t beam_width;
	const std::vector<std::string> vocab;
	const ExternalScorer ext_scorer;

	Decoder(int thread_count, int blank_id, int cutoff_top_n, int apostrophe_id, double nucleus_prob_per_timestep,
			double alpha, double beta, std::size_t beam_width, double lex_penalty, double min_tok_prob,
			double max_beam_score_deviation, char tok_sep, std::vector<std::string> vocab, char* lm_path,
			char* lexicon_path)
		: thread_count(thread_count)
		, blank_id(blank_id)
		, cutoff_top_n(cutoff_top_n)
		, vocab_size(vocab.size())
		, nucleus_prob_per_timestep(nucleus_prob_per_timestep)
		, min_tok_prob(std::exp(min_tok_prob))
		, max_beam_score_deviation(max_beam_score_deviation)
		, beam_width(beam_width)
		, vocab(vocab)
		, ext_scorer(tok_sep, apostrophe_id, alpha, beta, lex_penalty, lm_path, lexicon_path)
	{
	}

	fst::StdVectorFst* generate_hw_fst(const std::vector<std::vector<int>>& hotwords_id,
									   const std::vector<float>& hotwords_weight,
									   fst::StdVectorFst* hotwords_fst) const;

	template <typename T>
	void batch_decode(T* logits, int* ids, int* labels, int* timesteps, int* seq_len, int* seq_pos,
					  const int batch_size, const int max_seq_len, std::vector<std::vector<int>>& hotwords_id,
					  std::vector<float>& hotwords_weight, fst::StdVectorFst* hotwords_fst) const;

	/**
	 * @brief Decodes the provided logits using CTC Beam Search algorithm. This function is the main entry point
	 * from the Python bindings. Since `torch` passes the logits datapointer as a `long` type instead of a pointer,
	 * we need to use a wrapper function to convert the `long` type to the appropriate pointer type based on the
	 * logit bytes.
	 *
	 * @param logits The logits array pointer, which can be either a float or double pointer, depending on the logit
	 * bytes.
	 * @param logit_bytes The size of the logit type in bytes (either 4 for float or 8 for double).
	 * @param ids The sorted ids array pointer, which is an integer pointer.
	 * @param labels The labels array pointer, which is an integer pointer.
	 * @param timesteps The timesteps array pointer, which is an integer pointer.
	 * @param seq_len The sequence lengths array pointer, which is an integer pointer.
	 * @param seq_pos The sequence positions array pointer, which is an integer pointer.
	 * @param batch_size The number of batches to decode.
	 * @param max_seq_len The maximum sequence length for the batch.
	 * @param hotwords_id The hotwords ids vector, which is a vector of hotword token ids.
	 * @param hotwords_weight The hotwords weights vector, which is a vector of hotword token weights.
	 * @param hotwords_fst The hotwords finite state transducer, which is a pointer to a `fst::StdVectorFst` object.
	 *
	 * @note This function is used to decode the logits in a batch-wise manner, allowing for efficient decoding
	 * of multiple sequences at once. The logits should be in the shape of Batch x SeqLen x Vocab, containing the
	 * softmaxed probabilities in linear scale.
	 */
	void batch_decode_wrapper(long logits, int logit_bytes, long ids, long labels, long timesteps, long seq_len,
							  long seq_pos, const int batch_size, const int max_seq_len,
							  std::vector<std::vector<int>>& hotwords_id, std::vector<float>& hotwords_weight,
							  fst::StdVectorFst* hotwords_fst) const
	{
		if (logit_bytes == sizeof(float)) {
			this->batch_decode((float*)logits, (int*)ids, (int*)labels, (int*)timesteps, (int*)seq_len, (int*)seq_pos,
							   batch_size, max_seq_len, hotwords_id, hotwords_weight, hotwords_fst);
		} else if (logit_bytes == sizeof(double)) {
			this->batch_decode((double*)logits, (int*)ids, (int*)labels, (int*)timesteps, (int*)seq_len, (int*)seq_pos,
							   batch_size, max_seq_len, hotwords_id, hotwords_weight, hotwords_fst);
		} else {
			throw std::runtime_error("Invalid logit dtype. Expected floating point value of precision 32 or 64 bits.");
		}
	}

#ifndef NDEBUG
	// NOTE: This function is only for debugging purpose.
	template <typename T>
	void serial_decode(T* logits, int* ids, int* labels, int* timesteps, int* seq_len, int* seq_pos,
					   const int batch_size, const int max_seq_len, std::vector<std::vector<int>>& hotwords_id,
					   std::vector<float>& hotwords_weight, fst::StdVectorFst* hotwords_fst) const;

	void serial_decode_wrapper(long logits, int logit_bytes, long ids, long labels, long timesteps, long seq_len,
							   long seq_pos, const int batch_size, const int max_seq_len,
							   std::vector<std::vector<int>>& hotwords_id, std::vector<float>& hotwords_weight,
							   fst::StdVectorFst* hotwords_fst) const
	{
		if (logit_bytes == sizeof(float)) {
			this->serial_decode((float*)logits, (int*)ids, (int*)labels, (int*)timesteps, (int*)seq_len, (int*)seq_pos,
								batch_size, max_seq_len, hotwords_id, hotwords_weight, hotwords_fst);
		} else if (logit_bytes == sizeof(double)) {
			this->serial_decode((double*)logits, (int*)ids, (int*)labels, (int*)timesteps, (int*)seq_len, (int*)seq_pos,
								batch_size, max_seq_len, hotwords_id, hotwords_weight, hotwords_fst);
		} else {
			throw std::runtime_error("Invalid logit dtype. Expected floating point value of precision 32 or 64 bits.");
		}
	}
#endif // NDEBUG
};

/**
 * @brief Moves the clone nodes present in the source vector to the
 * start of the vector. This is done to avoid the unnecessary
 * expanding of the `source & deprecated` node.
 *
 * @param source The source vector from which the clone nodes are to be moved.
 *
 * @return void
 */
inline void
move_clones_to_start(std::vector<zctc::Node*>& source)
{
	for (int from_pos = 0, to_pos = 0; from_pos < source.size(); from_pos++) {
		if (!source[from_pos]->is_clone)
			continue;

		std::iter_swap(source.begin() + from_pos, source.begin() + to_pos);
		to_pos++;
	}
}

/**
 * @brief Removes the nodes from the source vector, based on the remove_ids
 * vector. To efficiently remove the elements, we'll swap the elements to be
 * removed to the end of the source vector and then erase the whole block
 * of elements.
 *
 * @param source The source vector from which the elements are to be removed.
 * @param remove_ids The vector containing the indices of the elements in the source vector to be removed.
 *
 * @return void
 */
inline void
remove_from_source(std::vector<zctc::Node*>& source, std::vector<int>& remove_ids)
{
	int to_pos = source.size() - 1;

	for (auto id = remove_ids.rbegin(); id != remove_ids.rend(); id++) {
		std::iter_swap(source.begin() + *id, source.begin() + to_pos);
		to_pos--;
	}

	source.erase(source.end() - remove_ids.size(), source.end());
	remove_ids.clear();
}

/**
 * @brief Decodes the provided logits using CTC Beam Search algorithm,
 * using the provided decoder configuration. The decoded labels, timesteps
 * and sequence lengths are written to the provided array pointers.
 *
 * @param decoder The decoder configuration to be used for decoding.
 * @param logits The logits array of shape Batch x SeqLen x Vocab, containing the softmaxed probabilities in linear
 * scale.
 * @param ids The sorted ids array of shape Batch x SeqLen x Vocab, containing the sorted indices of the logits at each
 * timestep.
 * @param label The labels array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded labels.
 * @param timestep The timesteps array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded timesteps.
 * @param seq_len The sequence length of the sample in the logits array excluding the padding.
 * @param max_seq_len The maximum sequence length of the sample in the logits array including the padding.
 * @param seq_pos The sequence position array of shape Batch x BeamWidth, to write the sequence starting position of the
 * decoded labels.
 * @param hotwords_fst The FST representing the hotwords, if any, to be used for decoding.
 *
 * @return int 0 on successful execution.
 */
template <typename T>
int
decode(const Decoder* decoder, T* logits, int* ids, int* label, int* timestep, const int seq_len, const int max_seq_len,
	   int* seq_pos, fst::StdVectorFst* hotwords_fst)
{
	bool is_blank, full_beam;
	int iter_val, pos_val;
	double nucleus_count, prob, max_beam_score, min_beam_score, beam_score;
	int *curr_id, *curr_l, *curr_t, *curr_p;
	zctc::Node* child;
	std::vector<int> writer_remove_ids;
	std::vector<zctc::Node*> prefixes0, prefixes1, more_confident_repeats;
	zctc::Node root(zctc::ROOT_ID, -1, 0.0, "<s>", nullptr);
	fst::SortedMatcher<fst::StdVectorFst> lexicon_matcher(decoder->ext_scorer.lexicon, fst::MATCH_INPUT);
	fst::SortedMatcher<fst::StdVectorFst> hotwords_matcher(hotwords_fst, fst::MATCH_INPUT);

	decoder->ext_scorer.initialise_start_states(&root, hotwords_fst);

	/**
	 * NOTE: For performance reasons, we initialise and reserve memory
	 * 		 for the prefixes.
	 */
	prefixes0.reserve(2 * decoder->beam_width);
	prefixes1.reserve(2 * decoder->beam_width);
	prefixes0.emplace_back(&root);

	for (int timestep = 0; timestep < seq_len; timestep++) {
		/**
		 * NOTE: Swap the reader and writer vectors, as per the timestep,
		 * 		 to avoid cleaning and copying the elements.
		 */
		std::vector<zctc::Node*>& reader = ((timestep % 2) == 0 ? prefixes0 : prefixes1);
		std::vector<zctc::Node*>& writer = ((timestep % 2) == 0 ? prefixes1 : prefixes0);

		nucleus_count = 0;
		iter_val = timestep * decoder->vocab_size;
		curr_id = ids + iter_val;
		full_beam = (reader.size() >= decoder->beam_width) && decoder->ext_scorer.enabled;
		move_clones_to_start(reader);

		if (full_beam) {
			/**
			 * NOTE: Parlance style of pruning the node extensions
			 * 		 based on their score.
			 */
			min_beam_score = std::numeric_limits<double>::max();
			for (zctc::Node* r_node : reader) {
				if (r_node->ovrl_score < min_beam_score)
					min_beam_score = r_node->ovrl_score;
			}

			min_beam_score += std::log(logits[iter_val + decoder->blank_id]) - std::abs(decoder->ext_scorer.beta);
		} else {
			min_beam_score = std::numeric_limits<double>::lowest();
		}

		for (int i = 0, index = 0; i < decoder->cutoff_top_n; i++, curr_id++) {
			index = *curr_id;
			// NOTE: Implicit type_casting from `T` to `double`.
			prob = logits[iter_val + index];

			if (prob < decoder->min_tok_prob)
				break;

			is_blank = index == decoder->blank_id;
			nucleus_count += prob;

			if (is_blank) {
				/**
				 * NOTE: Just update the blank probs of the node and
				 * 		 continue in case if the current is blank token.
				 */
				for (zctc::Node* r_node : reader) {
					r_node->b_prob = prob;
					if (!r_node->is_at_writer) {
						writer.emplace_back(r_node);
						r_node->is_at_writer = true;
					}
				}

				continue;
			}

			for (zctc::Node* r_node : reader) {
				/**
				 * NOTE: Parlance style will be just accumulating
				 * 		 the token probs, but we've included the blank
				 * 		 probs too, coz, there they'll be updating the
				 * 		 score of the node immediately after node extension,
				 * 		 but we update score only at the end of each timestep
				 * 		 parsing.
				 */
				if (full_beam && ((r_node->ovrl_score + std::log(prob)) < min_beam_score))
					break;

				child = r_node->extend_path(index, timestep, prob, decoder->vocab[index], writer, reader);

				/**
				 * NOTE: `nullptr` means the path extension was not done,
				 * 		 (ie) no new node was created, the probs were
				 * 		 accumulated within the current node, or the node
				 * 		 was cloned.
				 */
				if (child == nullptr)
					continue;

				/**
				 * NOTE: Only newly extended nodes from the `r_node` are
				 * 		 considered for external scoring. This is done once
				 * 		 per new node creation.
				 */
				decoder->ext_scorer.run_ext_scoring(child, &lexicon_matcher, hotwords_fst, &hotwords_matcher);
			}

			if (nucleus_count >= decoder->nucleus_prob_per_timestep)
				break;
		}

		pos_val = -1;
		max_beam_score = std::numeric_limits<double>::lowest();
		for (zctc::Node* w_node : writer) {
			/**
			 * NOTE: Updating the `score` and `ovrl_score` of the
			 * 		 nodes, considering the AM probs, KenLM probs,
			 * 		 lexicon penalty, hotword boosting values and
			 * 		 beta word penalty.
			 */
			pos_val++;

			beam_score = w_node->update_score(timestep, more_confident_repeats);

			if (w_node->is_deprecated) {
				writer_remove_ids.emplace_back(pos_val);
				continue;
			}
			/**
			 * NOTE: Doing the update step here, to avoid
			 * 		 the current timestep's repeat token prob
			 * 		 of the node, getting included with a
			 * 		 different symbol that is getting extended
			 * 		 in this timestep, like,
			 *
			 * 		-->        	a - In this case, the probs will be acc to the curr node itself.
			 * 						If the prev node has a most recent blank too, then new node
			 * 						will also be created and the path will be extended.
			 * 		|
			 * 	a ------> (blank) - In this case, the probs will be acc to the curr node itself.
			 * 		|
			 * 		--> 	    b - In this case, a new node is created and the path is extended.
			 */
			if (beam_score > max_beam_score)
				max_beam_score = beam_score;
		}

		remove_from_source(writer, writer_remove_ids);
		for (zctc::Node* repeat_node : more_confident_repeats) {
			writer.emplace_back(repeat_node);
		}
		more_confident_repeats.clear();

		reader.clear();
		if (writer.size() <= decoder->beam_width)
			continue;

		pos_val = 0;
		beam_score = max_beam_score + decoder->max_beam_score_deviation;
		for (zctc::Node* w_node : writer) {
			if (w_node->ovrl_score < beam_score)
				writer_remove_ids.emplace_back(pos_val);

			pos_val++;
		}
		remove_from_source(writer, writer_remove_ids);
		if (writer.size() <= decoder->beam_width)
			continue;

		std::nth_element(writer.begin(), writer.begin() + decoder->beam_width, writer.end(),
						 Decoder::descending_compare);
		// TODO: Try `resize()` instead of `erase()`, to avoid memory issue during benchmarking.
		writer.erase(writer.begin() + decoder->beam_width, writer.end());
	}

	std::vector<zctc::Node*>& reader = ((seq_len % 2) == 0 ? prefixes0 : prefixes1);
	std::sort(reader.begin(), reader.end(), Decoder::descending_compare);

	iter_val = 1;
	curr_p = seq_pos;
	for (zctc::Node* r_node : reader) {

		curr_t = timestep + ((max_seq_len * iter_val) - 1);
		curr_l = label + ((max_seq_len * iter_val) - 1);
		pos_val = max_seq_len;

		while (r_node->id != zctc::ROOT_ID) {

			*curr_l = r_node->id;
			*curr_t = r_node->ts;

			curr_l--;
			curr_t--;
			pos_val--;

			r_node = r_node->parent;
		}

		*curr_p = pos_val;
		iter_val++;
		curr_p++;
	}

	return 0;
}

} // namespace zctc

/* ---------------------------------------------------------------------------- */

/**
 * @brief Compares the two nodes based on the `ovrl_score` and `seq_length`,
 * 		  considering higher `ovrl_score` and lower `seq_length` as the best.
 *
 * @param x The first node to be compared.
 * @param y The second node to be compared.
 *
 * @return `true` If the first node is better than the second node.
 * @return `false` If the second node is better than the first node.
 */
bool
zctc::Decoder::descending_compare(zctc::Node* x, zctc::Node* y)
{
	// NOTE: If probabilities are same, then we'll consider shorter sequences.
	return x->ovrl_score > y->ovrl_score; // ? x->ovrl_score > y->ovrl_score : x->seq_length < y->seq_length;
}

/**
 * @brief Concurrently decodes the provided batch of logits and its supporting
 *  	  arrays using CTC Beam Search algorithm, using the provided decoder
 * 		  configuration. The decoded labels, timesteps and sequence positions
 * 		  are written to the provided array pointers.
 *
 * @tparam T The type of the logits array.
 * @param batch_log_logits The batch of logits array of shape Batch x SeqLen x Vocab, containing the softmaxed
 * probabilities in linear scale.
 * @param batch_sorted_ids The batch of sorted ids array of shape Batch x SeqLen x Vocab, containing the sorted indices
 * of the logits at each timestep.
 * @param batch_labels The batch of labels array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded labels.
 * @param batch_timesteps The batch of timesteps array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded
 * timesteps.
 * @param batch_seq_len The batch of sequence length of the samples in the logits array excluding the padding.
 * @param batch_seq_pos The batch of sequence position array of shape Batch x BeamWidth, to write the sequence starting
 * position of the decoded labels.
 * @param batch_size The number of samples in the batch.
 * @param max_seq_len The maximum sequence length of the samples in the logits array including the padding.
 * @param hotwords Vector of hotword tokens to consider for hotword boosting.
 * @param hotwords_weight Vector of hotword weights to consider for hotword boosting.
 *
 * @return void
 */
template <typename T>
void
zctc::Decoder::batch_decode(T* logits, int* ids, int* labels, int* timesteps, int* seq_len, int* seq_pos,
							const int batch_size, const int max_seq_len, std::vector<std::vector<int>>& hotwords_id,
							std::vector<float>& hotwords_weight, fst::StdVectorFst* hotwords_fst) const
{
	ThreadPool pool(std::min(this->thread_count, batch_size));
	std::vector<std::future<int>> results;
	bool free_hw_fst = false;

	if (!hotwords_id.empty()) {
		if (hotwords_fst == nullptr) {
			hotwords_fst = new fst::StdVectorFst();
			free_hw_fst = true;
		} else {
			/**
			 * NOTE: The reason for cloning `hotwords_fst` is to avoid
			 * 		 unncessary overwriting of the parameterly passed
			 * 		 `hotwords_fst`.
			 */
			hotwords_fst = new fst::StdVectorFst(*hotwords_fst);
			free_hw_fst = true;
		}
		populate_hotword_fst(hotwords_fst, hotwords_id, hotwords_weight);
	}

	for (int i = 0, ip_pos = 0, op_pos = 0, s_p = 0; i < batch_size; i++) {
		ip_pos = i * max_seq_len * this->vocab_size;
		op_pos = i * this->beam_width * max_seq_len;
		s_p = i * this->beam_width;

		results.emplace_back(pool.enqueue(zctc::decode<T>, this, logits + ip_pos, ids + ip_pos, labels + op_pos,
										  timesteps + op_pos, *(seq_len + i), max_seq_len, seq_pos + s_p,
										  hotwords_fst));
	}

	for (auto&& result : results)
		if (result.get() != 0)
			throw std::runtime_error("Unexpected error occured during execution");

	if (free_hw_fst)
		delete hotwords_fst;
}

/**
 * @brief Populates the hotword FST with the provided hotwords and their weights.
 *
 * @param hotwords_id Vector of hotword tokens to consider for hotword boosting.
 * @param hotwords_weight Vector of hotword weights to consider for hotword boosting.
 *
 * @return fst::StdVectorFst The populated hotword FST.
 */
fst::StdVectorFst*
zctc::Decoder::generate_hw_fst(const std::vector<std::vector<int>>& hotwords_id,
							   const std::vector<float>& hotwords_weight, fst::StdVectorFst* hotwords_fst) const
{
	if (hotwords_fst == nullptr)
		hotwords_fst = new fst::StdVectorFst();

	zctc::populate_hotword_fst(hotwords_fst, hotwords_id, hotwords_weight);

	return hotwords_fst;
}

#ifndef NDEBUG

/**
 * @brief Serially decodes the provided batch of logits and its supporting
 *  	  arrays using CTC Beam Search algorithm, using the provided decoder
 * 		  configuration. The decoded labels, timesteps and sequence positions
 * 		  are written to the provided array pointers.
 *
 * @note This function is only for debugging purpose. It will not be inlcuded
 * 		 in the release build.
 *
 * @tparam T The type of the logits array.
 * @param batch_log_logits The batch of logits array of shape Batch x SeqLen x Vocab, containing the softmaxed
 * probabilities in linear scale.
 * @param batch_sorted_ids The batch of sorted ids array of shape Batch x SeqLen x Vocab, containing the sorted indices
 * of the logits at each timestep.
 * @param batch_labels The batch of labels array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded labels.
 * @param batch_timesteps The batch of timesteps array of shape Batch x BeamWidth x MaxSeqLen, to write the decoded
 * timesteps.
 * @param batch_seq_len The batch of sequence length of the samples in the logits array excluding the padding.
 * @param batch_seq_pos The batch of sequence position array of shape Batch x BeamWidth, to write the sequence starting
 * position of the decoded labels.
 * @param batch_size The number of samples in the batch.
 * @param max_seq_len The maximum sequence length of the samples in the logits array including the padding.
 * @param hotwords Vector of hotword tokens to consider for hotword boosting.
 * @param hotwords_weight Vector of hotword weights to consider for hotword boosting.
 *
 * @return void
 */
template <typename T>
void
zctc::Decoder::serial_decode(T* logits, int* ids, int* labels, int* timesteps, int* seq_len, int* seq_pos,
							 const int batch_size, const int max_seq_len, std::vector<std::vector<int>>& hotwords_id,
							 std::vector<float>& hotwords_weight, fst::StdVectorFst* hotwords_fst) const
{
	bool free_hw_fst = false;

	if (!hotwords_id.empty()) {
		if (hotwords_fst == nullptr) {
			hotwords_fst = new fst::StdVectorFst();
			free_hw_fst = true;
		} else {
			/**
			 * NOTE: The reason for cloning `hotwords_fst` is to avoid
			 * 		 unncessary overwriting of the parameterly passed
			 * 		 `hotwords_fst`.
			 */
			hotwords_fst = new fst::StdVectorFst(*hotwords_fst);
			free_hw_fst = true;
		}
		populate_hotword_fst(hotwords_fst, hotwords_id, hotwords_weight);
	}

	for (int i = 0, ip_pos = 0, op_pos = 0, s_p = 0; i < batch_size; i++) {
		ip_pos = i * max_seq_len * this->vocab_size;
		op_pos = i * this->beam_width * max_seq_len;
		s_p = i * this->beam_width;

		zctc::decode<T>(this, logits + ip_pos, ids + ip_pos, labels + op_pos, timesteps + op_pos, *(seq_len + i),
						max_seq_len, seq_pos + s_p, hotwords_fst);
	}

	if (free_hw_fst)
		delete hotwords_fst;
}

#endif // NDEBUG

#endif // _ZCTC_DECODER_H
