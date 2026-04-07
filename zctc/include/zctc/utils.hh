#ifndef _ZCTC_CONSTANTS_H
#define _ZCTC_CONSTANTS_H

#include <cmath>

namespace zctc {

static constexpr int ROOT_ID = -1;
static constexpr float LOG_A_OF_B = std::log10(std::exp(1.0f));

/**
 * @brief Calculate the hotword score based on the match length, final state and token score of the hotword.
 *        The hotword score is calculated using a quadratic function of the match length, and is scaled
 *        by the token score of the hotword.
 *
 * @param hw_match_len The length of the matched hotword tokens.
 * @param hw_total_len The total length of the actual hotword.
 * @param hw_score Total boosting score of the hotword.
 *
 * @return T The calculated hotword score for the matched hotword tokens.
 */
template <typename T>
T
quadratic_hw_score(int hw_match_len, int hw_total_len, T hw_score)
{
	return std::pow(hw_match_len / hw_total_len, 2) * hw_score;
}

}

#endif // _ZCTC_CONSTANTS_H
