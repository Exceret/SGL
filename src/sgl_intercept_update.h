#ifndef SGL_INTERCEPT_UPDATE_HPP
#define SGL_INTERCEPT_UPDATE_HPP

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    inline double stable_logistic_probability(
    const double eta
    ) noexcept
    {
        if (eta >= 0.0)
        {
            const double exp_neg_eta =
                std::exp(-eta);
            return 1.0 /
                   (1.0 + exp_neg_eta);
        }
        const double exp_eta =
            std::exp(eta);
        return exp_eta /
               (1.0 + exp_eta);
    }


    /*
     * Unnormalized linear-model intercept gradient:
     *
     *   dL/db0 = sum_i (eta_i - y_i)
     */
    inline double linear_intercept_gradient(
        const arma::mat& eta,
              const arma::mat& y
    ) noexcept
    {
        const arma::uword n =
            eta.n_rows;
        const double *eta_ptr =
            eta.memptr();
        const double *y_ptr =
            y.memptr();
        double gradient = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            gradient +=
                eta_ptr[i] -
                y_ptr[i];
        }
        return gradient;
    }


    /*
     * Unnormalized Logistic intercept gradient:
     *
     *   dL/db0 = sum_i (plogis(eta_i) - y_i)
     */
    inline double logistic_intercept_gradient(
        const arma::mat& eta,
        const arma::mat& y
    ) noexcept
    {
        const arma::uword n =
            eta.n_rows;
        const double *eta_ptr =
            eta.memptr();
        const double *y_ptr =
            y.memptr();
        double gradient = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            const double probability =
                stable_logistic_probability(
                    eta_ptr[i]
                );
            gradient +=
                probability -
                y_ptr[i];
        }
        return gradient;
    }


    /*
     * Unnormalized Logistic intercept Hessian:
     *
     *   d2L/db0^2 =
     *       sum_i p_i * (1 - p_i)
     */
    inline double logistic_intercept_hessian(
        const arma::mat& eta
    ) noexcept
    {
        const arma::uword n =
            eta.n_rows;
        const double *eta_ptr =
            eta.memptr();
        double hessian = 0.0;
        for (arma::uword i = 0; i < n; ++i)
        {
            const double probability =
                stable_logistic_probability(
                    eta_ptr[i]
                );
            hessian +=
                probability *
                (1.0 - probability);
        }
        return hessian;
    }


    /*
     * Intercept-only state update:
     *
     *   intercept_new = intercept_old + delta_intercept
     *   eta_new       = eta_old + delta_intercept
     *
     * Both objects are updated in place.
     */
    inline void update_intercept_eta_inplace(
        arma::mat& intercept,
        arma::mat& eta,
        const double delta_intercept
    ) noexcept
    {
        intercept(0, 0) +=
            delta_intercept;
        const arma::uword n =
            eta.n_rows;
        double *eta_ptr =
            eta.memptr();
        for (arma::uword i = 0; i < n; ++i)
        {
            eta_ptr[i] +=
                delta_intercept;
        }
    }

} // namespace sgl

#endif