#ifndef SGL_LINEAR_PREDICTOR_HPP
#define SGL_LINEAR_PREDICTOR_HPP

#include <RcppArmadillo.h>

namespace sgl
{

    /*
     * Recompute the linear predictor in place:
     *
     *   eta <- intercept + X %*% beta
     *
     * X         : n x p
     * beta      : p x 1
     * intercept : 1 x 1
     * eta       : n x 1
     *
     * The column-wise accumulation order is deterministic.
     * No X %*% beta temporary matrix is created.
     */
    inline void initialize_linear_predictor_inplace(
    arma::mat& eta,
    const arma::mat& X,
    const arma::mat& beta,
    const arma::mat& intercept
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const arma::uword p = X.n_cols;
        double *eta_ptr = eta.memptr();
        const double *beta_ptr = beta.memptr();
        const double intercept_value =
            intercept(0, 0);
        for (arma::uword i = 0; i < n; ++i)
        {
            eta_ptr[i] = intercept_value;
        }
        for (arma::uword j = 0; j < p; ++j)
        {
            const double beta_j = beta_ptr[j];
            if (beta_j == 0.0)
            {
                continue;
            }
            const double *x_ptr =
                X.colptr(j);
            for (arma::uword i = 0; i < n; ++i)
            {
                eta_ptr[i] += x_ptr[i] * beta_j;
            }
        }
    }


    /*
     * Update the linear predictor from old and new model states:
     *
     *   eta_new =
     *       eta_old +
     *       X %*% (beta_new - beta_old) +
     *       (intercept_new - intercept_old)
     *
     * beta_old and beta_new : p x 1
     * intercept_old/new     : 1 x 1
     * eta                   : n x 1
     *
     * All updates are performed in place.
     */
    inline void update_linear_predictor_state_inplace(
        arma::mat& eta,
             const arma::mat& X,
             const arma::mat& beta_old,
             const arma::mat& beta_new,
             const arma::mat& intercept_old,
             const arma::mat& intercept_new
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const arma::uword p = X.n_cols;
        double *eta_ptr = eta.memptr();
        const double *beta_old_ptr =
            beta_old.memptr();
        const double *beta_new_ptr =
            beta_new.memptr();
        const double delta_intercept =
            intercept_new(0, 0) -
            intercept_old(0, 0);
        if (delta_intercept != 0.0)
        {
            for (arma::uword i = 0; i < n; ++i)
            {
                eta_ptr[i] += delta_intercept;
            }
        }
        for (arma::uword j = 0; j < p; ++j)
        {
            const double delta_beta =
                beta_new_ptr[j] -
                beta_old_ptr[j];
            if (delta_beta == 0.0)
            {
                continue;
            }
            const double *x_ptr =
                X.colptr(j);
            for (arma::uword i = 0; i < n; ++i)
            {
                eta_ptr[i] += x_ptr[i] * delta_beta;
            }
        }
    }

} // namespace sgl

#endif