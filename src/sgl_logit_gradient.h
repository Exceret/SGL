#ifndef SGL_LOGIT_GRADIENT_H
#define SGL_LOGIT_GRADIENT_H

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    /*
     * Gradient of the mean Logistic negative log-likelihood:
     *
     *   L(beta) = sum_i [
     *       log(1 + exp(eta_i)) - y_i * eta_i
     *   ]
     *
     *   dL/dbeta = X^T %*% (p - y)
     *
     * where:
     *
     *   p_i = plogis(eta_i)
     *
     * All vector-like objects are n x 1 or p x 1 matrices.
     *
     * The calculation is fused:
     *   - no probability vector is allocated;
     *   - no residual vector is allocated;
     *   - gradient is returned as a p x 1 matrix.
     */
    inline void logistic_nll_gradient(
    arma::mat& gradient,
    const arma::mat& X,
    const arma::mat& eta,
    const arma::mat& y
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const arma::uword p = X.n_cols;
        const double *eta_ptr = eta.memptr();
        const double *y_ptr = y.memptr();
        double *gradient_ptr = gradient.memptr();
        for (arma::uword j = 0; j < p; ++j)
        {
            const double *x_ptr = X.colptr(j);
            double value = 0.0;
            for (arma::uword i = 0; i < n; ++i)
            {
                const double eta_i = eta_ptr[i];
                const double y_i = y_ptr[i];
                double probability;
                if (eta_i >= 0.0)
                {
                    const double exp_neg_eta = std::exp(-eta_i);
                    probability =
                        1.0 / (1.0 + exp_neg_eta);
                }
                else
                {
                    const double exp_eta = std::exp(eta_i);
                    probability =
                        exp_eta / (1.0 + exp_eta);
                }
                value += x_ptr[i] * (probability - y_i);
            }
            gradient_ptr[j] =
                value /
                static_cast<double>(n);
        }
    }

} // namespace sgl

#endif