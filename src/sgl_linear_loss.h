#ifndef SGL_LINEAR_LOSS_H
#define SGL_LINEAR_LOSS_H

#include <RcppArmadillo.h>

namespace sgl
{

    /*
     * Unnormalized least-squares objective:
     *
     *   L(eta, y) = 1/2 * sum_i (eta_i - y_i)^2
     *
     * eta and y are n x 1 matrices.
     */
    inline double linear_squared_loss(
    const arma::mat& eta,
    const arma::mat& y
    ) noexcept
    {
        const arma::uword n = eta.n_rows;
        const double *eta_ptr = eta.memptr();
        const double *y_ptr = y.memptr();
        double loss = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            const double residual =
                eta_ptr[i] - y_ptr[i];
            loss += 0.5 * residual * residual;
        }
        return loss /
               static_cast<double>(n);
    }

} // namespace sgl

#endif