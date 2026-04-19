#ifndef _ZCTC_CONSTANTS_H
#define _ZCTC_CONSTANTS_H

#include <cmath>

namespace zctc {

static constexpr int ROOT_ID = -1;
static constexpr float LOG_A_OF_B = std::log10(std::exp(1.0f));

/**
 * @brief Calculate the hotword score based on the completion ratio of the word and it's respective
 * 		  hotword score. The hotword score is calculated using quadratic function and is scaled
 * 	  	  by the token score of the hotword.
 *
 * @param hw_completion_ratio The completion ratio of the hotword tokens matched so far,
 * 							  which is calculated as (hw_match_len / hw_total_len).
 * @param hw_score Total boosting score of the hotword.
 *
 * @return T The calculated hotword score for the matched hotword tokens.
 */
template <typename T>
T
quadratic_hw_score(T hw_completion_ratio, T hw_score)
{
	return std::pow(hw_completion_ratio, (T)2) * hw_score;
}

/**
 * @brief Calculate the hotword score based on the completion ratio of the word and it's respective
 * 		  hotword score. The hotword score is calculated using linear function and is scaled
 *        by the token score of the hotword.
 *
 * @param hw_completion_ratio The completion ratio of the hotword tokens matched so far,
 * 							  which is calculated as (hw_match_len / hw_total_len).
 * @param hw_score Total boosting score of the hotword.
 *
 * @return T The calculated hotword score for the matched hotword tokens.
 */
template <typename T>
T
linear_hw_score(T hw_completion_ratio, T hw_score)
{
	return hw_completion_ratio * hw_score;
}

}

#endif // _ZCTC_CONSTANTS_H
