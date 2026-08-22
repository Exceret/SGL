#ifndef SGL_LINEAR_GRADIENT_H
#define SGL_LINEAR_GRADIENT_H

#include <RcppArmadillo.h>

namespace sgl
{

    /*
     * Gradient of the mean least-squares objective:
     *
     *   L(beta) = 1/(2n) * sum_i (eta_i - y_i)^2
     *
     *   dL/dbeta = X^T %*% (eta - y) / n
     */
    inline void linear_gradient(
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
                const double residual =
                    eta_ptr[i] - y_ptr[i];
                value += x_ptr[i] * residual;
            }
            gradient_ptr[j] =
                value /
                static_cast<double>(n);
        }
    }

} // namespace sgl

#endif