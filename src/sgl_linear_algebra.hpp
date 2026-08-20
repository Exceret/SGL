#ifndef SGL_LINEAR_ALGEBRA_HPP
#define SGL_LINEAR_ALGEBRA_HPP

#include <RcppArmadillo.h>

namespace sgl
{

    /*
     * All vector-like objects use an n x 1 arma::mat representation.
     *
     * eta      : n x 1
     * beta     : p x 1
     * residual : n x 1
     * gradient : p x 1
     */

    inline void update_eta_inplace(
    arma::mat& eta,
    const arma::mat& X,
    const arma::mat& delta_beta
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const arma::uword p = X.n_cols;
        double *eta_ptr = eta.memptr();
        const double *delta_ptr = delta_beta.memptr();
        for (arma::uword j = 0; j < p; ++j)
        {
            const double delta = delta_ptr[j];
            if (delta == 0.0)
            {
                continue;
            }
            const double *x_ptr = X.colptr(j);
            for (arma::uword i = 0; i < n; ++i)
            {
                eta_ptr[i] += x_ptr[i] * delta;
            }
        }
    }


    inline void update_eta_group_inplace(
        arma::mat& eta,
             const arma::mat& X,
             const arma::mat& delta_beta,
             const arma::uword first_col,
             const arma::uword last_col
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        double *eta_ptr = eta.memptr();
        const double *delta_ptr = delta_beta.memptr();
        for (arma::uword j = first_col; j <= last_col; ++j)
        {
            const double delta = delta_ptr[j];
            if (delta == 0.0)
            {
                continue;
            }
            const double *x_ptr = X.colptr(j);
            for (arma::uword i = 0; i < n; ++i)
            {
                eta_ptr[i] += x_ptr[i] * delta;
            }
        }
    }


    inline void gradient_xt_residual(
        arma::mat& gradient,
        const arma::mat& X,
        const arma::mat& residual
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const arma::uword p = X.n_cols;
        double *gradient_ptr = gradient.memptr();
        const double *residual_ptr = residual.memptr();
        for (arma::uword j = 0; j < p; ++j)
        {
            const double *x_ptr = X.colptr(j);
            double value = 0.0;
            for (arma::uword i = 0; i < n; ++i)
            {
                value += x_ptr[i] * residual_ptr[i];
            }
            gradient_ptr[j] = value;
        }
    }


    inline void gradient_group_xt_residual(
        arma::mat& gradient,
        const arma::mat& X,
        const arma::mat& residual,
        const arma::uword first_col,
        const arma::uword last_col
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const double *residual_ptr = residual.memptr();
        double *gradient_ptr = gradient.memptr();
        for (arma::uword j = first_col; j <= last_col; ++j)
        {
            const double *x_ptr = X.colptr(j);
            double value = 0.0;
            for (arma::uword i = 0; i < n; ++i)
            {
                value += x_ptr[i] * residual_ptr[i];
            }
            gradient_ptr[j] = value;
        }
    }

} // namespace sgl

#endif