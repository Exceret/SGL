#ifndef SGL_LOGIT_H
#define SGL_LOGIT_H

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    /*
     * Numerically stable logistic probability:
     *
     *   p = 1 / (1 + exp(-eta))
     *
     * eta and probability are both represented as n x 1 matrices.
     */
    inline void logistic_probability(
    arma::mat& probability,
    const arma::mat& eta
    ) noexcept
    {
        const arma::uword n = eta.n_rows;
        const double *eta_ptr = eta.memptr();
        double *probability_ptr = probability.memptr();
        for (arma::uword i = 0; i < n; ++i)
        {
            const double value = eta_ptr[i];
            if (value >= 0.0)
            {
                const double exp_neg = std::exp(-value);
                probability_ptr[i] =
                    1.0 / (1.0 + exp_neg);
            }
            else
            {
                const double exp_pos = std::exp(value);
                probability_ptr[i] =
                    exp_pos / (1.0 + exp_pos);
            }
        }
    }

} // namespace sgl

#endif