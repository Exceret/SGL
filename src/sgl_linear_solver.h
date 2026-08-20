#ifndef SGL_LINEAR_SOLVER_HPP
#define SGL_LINEAR_SOLVER_HPP

#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>

#include "sgl_group_prox.hpp"

namespace sgl
{

    /*
     * 对一次 proximal-gradient beta 更新执行：
     *
     *   candidate_j =
     *       beta_j - step_size * gradient_j
     *
     *   u_j =
     *       S(candidate_j, step_size * lambda * alpha)
     *
     *   beta_new_g =
     *       max(
     *           1 -
     *           step_size * lambda * (1 - alpha) *
     *           group_weight_g / ||u_g||_2,
     *           0
     *       ) * u_g
     *
     * 同时增量更新：
     *
     *   eta <- eta + X %*% (beta_new - beta_old)
     *
     * gradient 必须是当前 eta 对应的 beta 梯度。
     *
     * 函数返回本次 beta 更新的最大绝对变化量。
     */
    inline double sparse_group_proximal_gradient_update_eta_inplace(
    arma::mat& beta,
    arma::mat& eta,
    const arma::mat& X,
    const arma::mat& gradient,
    const GroupLayout& layout,
    const arma::mat& group_weight,
    const double step_size,
    const double lambda_l1,
    const double lambda_group
    ) noexcept
    {
        const arma::uword n_groups =
            group_weight.n_rows;
        const double *gradient_ptr =
            gradient.memptr();
        const double *weight_ptr =
            group_weight.memptr();
        double *beta_ptr =
            beta.memptr();
        double *eta_ptr =
            eta.memptr();
        double max_change = 0.0;
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            const arma::uword begin =
                layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            const double group_threshold =
                lambda_group * weight_ptr[g];
            double squared_norm = 0.0;
            /*
             * 第一遍：只读取 beta，计算近端前的组范数。
             */
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                const double candidate =
                    beta_ptr[j] -
                    step_size * gradient_ptr[j];
                const double value =
                    group_soft_threshold(
                        candidate,
                        lambda_l1
                    );
                squared_norm +=
                    value * value;
            }
            double factor = 0.0;
            if (squared_norm > 0.0)
            {
                const double group_norm =
                    std::sqrt(squared_norm);
                if (group_norm > group_threshold)
                {
                    factor =
                        1.0 -
                        group_threshold /
                        group_norm;
                }
            }
            /*
             * 第二遍：更新 beta，并增量更新 eta。
             */
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                const double old_beta =
                    beta_ptr[j];
                const double candidate =
                    old_beta -
                    step_size * gradient_ptr[j];
                const double l1_value =
                    group_soft_threshold(
                        candidate,
                        lambda_l1
                    );
                const double new_beta =
                    factor * l1_value;
                const double delta_beta =
                    new_beta - old_beta;
                beta_ptr[j] = new_beta;
                const double absolute_change =
                    std::abs(delta_beta);
                if (absolute_change > max_change)
                {
                    max_change = absolute_change;
                }
                if (delta_beta == 0.0)
                {
                    continue;
                }
                const double *x_ptr =
                    X.colptr(j);
                for (arma::uword i = 0;
                        i < X.n_rows;
                        ++i)
                {
                    eta_ptr[i] +=
                        x_ptr[i] * delta_beta;
                }
            }
        }
        return max_change;
    }


    /*
     * 计算线性 SGL 目标函数：
     *
     *   1/(2n) * ||y - eta||^2
     *   + lambda * [
     *       alpha * ||beta||_1
     *       + (1-alpha) * sum_g w_g ||beta_g||_2
     *     ]
     *
     * eta 和 y 必须是 n x 1。
     */
    inline double linear_sparse_group_objective(
        const arma::mat& beta,
              const arma::mat& eta,
              const arma::mat& y,
              const GroupLayout& layout,
              const arma::mat& group_weight,
              const double lambda,
              const double alpha
    ) noexcept
    {
        const arma::uword n =
            eta.n_rows;
        const double *beta_ptr =
            beta.memptr();
        const double *eta_ptr =
            eta.memptr();
        const double *y_ptr =
            y.memptr();
        const double *weight_ptr =
            group_weight.memptr();
        double squared_loss = 0.0;
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            const double residual =
                eta_ptr[i] - y_ptr[i];
            squared_loss +=
                residual * residual;
        }
        double l1_norm = 0.0;
        for (arma::uword j = 0;
                j < beta.n_rows;
                ++j)
        {
            l1_norm +=
                std::abs(beta_ptr[j]);
        }
        double weighted_group_norm = 0.0;
        for (arma::uword g = 0;
                g < group_weight.n_rows;
                ++g)
        {
            const arma::uword begin =
                layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            double group_squared_norm = 0.0;
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                group_squared_norm +=
                    beta_ptr[j] * beta_ptr[j];
            }
            weighted_group_norm +=
                weight_ptr[g] *
                std::sqrt(group_squared_norm);
        }
        const double n_double =
            static_cast<double>(n);
        return 0.5 * squared_loss / n_double +
               lambda * (
                   alpha * l1_norm +
                   (1.0 - alpha) *
                   weighted_group_norm
               );
    }

} // namespace sgl

#endif