#ifndef SGL_LOGIT_LOSS_H
#define SGL_LOGIT_LOSS_H

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    inline double stable_log1pexp(
    const double x
    ) noexcept
    {
        if (x > 0.0)
    {
        return x + std::log1p(std::exp(-x));
        }
        return std::log1p(std::exp(x));
    }


    /*
     * Bernoulli logistic negative log-likelihood:
     *
     *   L(eta, y) =
     *       sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ]
     *
     * eta and y are n x 1 matrices.
     */
    inline double logistic_negative_log_likelihood(
        const arma::mat& eta,
              const arma::mat& y
    ) noexcept
    {
        const arma::uword n = eta.n_rows;
        const double *eta_ptr = eta.memptr();
        const double *y_ptr = y.memptr();
        double value = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            const double eta_i = eta_ptr[i];
            value +=
                stable_log1pexp(eta_i) -
                y_ptr[i] * eta_i;
        }
        return value;
    }

} // namespace sgl

#endif