#ifndef SGL_FISTA_H
#define SGL_FISTA_H

#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>

namespace sgl
{
    /*
     * FISTA:
     *
     * t_{k+1} =
     *     (1 + sqrt(1 + 4 t_k^2)) / 2
     */
    inline double fista_next_t(
    const double t
    ) noexcept
    {
        return 0.5 *
               (
               1.0 +
               std::sqrt(
               1.0 + 4.0 * t * t
               )
        );
    }

    /*
     * 将 proximal 更新后的 beta/eta 作为
     * 当前解，并生成下一轮 extrapolated 状态：
     *
     * beta_extrapolated =
     *     beta +
     *     coefficient * (beta - beta_old)
     *
     * eta_extrapolated =
     *     eta +
     *     coefficient * (eta - eta_old)
     *
     * 返回 beta 的最大实际变化量。
     */
    inline double fista_commit_and_extrapolate_inplace(
        arma::mat& beta,
             arma::mat& beta_extrapolated,
             arma::mat& eta,
             arma::mat& eta_extrapolated,
             const double coefficient
    ) noexcept
    {
        double max_change = 0.0;
        double *beta_ptr =
            beta.memptr();
        double *beta_extrapolated_ptr =
            beta_extrapolated.memptr();
        for (arma::uword j = 0;
                j < beta.n_elem;
                ++j)
        {
            const double old_beta =
                beta_ptr[j];
            const double new_beta =
                beta_extrapolated_ptr[j];
            const double delta_beta =
                new_beta - old_beta;
            beta_ptr[j] =
                new_beta;
            beta_extrapolated_ptr[j] =
                new_beta +
                coefficient * delta_beta;
            max_change =
                std::max(
                    max_change,
                    std::abs(delta_beta)
                );
        }
        double *eta_ptr =
            eta.memptr();
        double *eta_extrapolated_ptr =
            eta_extrapolated.memptr();
        for (arma::uword i = 0;
                i < eta.n_elem;
                ++i)
        {
            const double old_eta =
                eta_ptr[i];
            const double new_eta =
                eta_extrapolated_ptr[i];
            const double delta_eta =
                new_eta - old_eta;
            eta_ptr[i] =
                new_eta;
            eta_extrapolated_ptr[i] =
                new_eta +
                coefficient * delta_eta;
        }
        return max_change;
    }

    /*
     * intercept 的 FISTA 状态更新。
     */
    inline double fista_commit_scalar_and_extrapolate_inplace(
        arma::mat& intercept,
        arma::mat& intercept_extrapolated,
        const double coefficient
    ) noexcept
    {
        const double old_intercept =
            intercept(0, 0);
        const double new_intercept =
            intercept_extrapolated(0, 0);
        const double delta_intercept =
            new_intercept - old_intercept;
        intercept(0, 0) =
            new_intercept;
        intercept_extrapolated(0, 0) =
            new_intercept +
            coefficient * delta_intercept;
        return std::abs(
                   delta_intercept
               );
    }

} // namespace sgl

#endif