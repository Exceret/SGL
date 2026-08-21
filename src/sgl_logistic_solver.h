#ifndef SGL_LOGISTIC_SOLVER_H
#define SGL_LOGISTIC_SOLVER_H

#include <RcppArmadillo.h>
#include <cmath>

#include "sgl_group_prox.h"
#include "sgl_intercept_update.h"

namespace sgl
{

    /*
     * 计算 Logistic 梯度：
     *
     *   gradient = X^T %*% (plogis(eta) - y) / n
     *
     * 不创建 probability 或 residual 临时向量。
     */
    inline void logistic_gradient_inplace(
    arma::mat& gradient,
    const arma::mat& X,
    const arma::mat& eta,
    const arma::mat& y
    ) noexcept
    {
        const arma::uword n = X.n_rows;
        const double *eta_ptr =
            eta.memptr();
        const double *y_ptr =
            y.memptr();
        arma::vec residual(n);
        for (arma::uword i = 0; i < n; ++i)
        {
            residual[i] =
                stable_logistic_probability(
                    eta_ptr[i]
                ) - y_ptr[i];
        }
        const double inverse_n =
            1.0 / static_cast<double>(n);
        // gradient = X.t() * residual / n
        gradient =
            X.t() * residual;
        gradient *= inverse_n;
    }


    /*
     * 对 Logistic proximal-gradient 执行一次 beta 更新，
     * 并同步增量更新 eta。
     *
     * gradient 已经是平均梯度：
     *
     *   X^T(p-y) / n
     *
     * lambda_l1：
     *
     *   step_size * lambda * alpha
     *
     * lambda_group：
     *
     *   step_size * lambda * (1-alpha)
     */
    inline double logistic_sparse_group_proximal_gradient_update_eta_inplace(
        arma::mat& beta,
             arma::mat& eta,
             const arma::mat& X,
             const arma::mat& gradient,
             const GroupLayout& layout,
             const arma::mat& group_weight,
             const double lambda_l1,
             const double lambda_group,
             const double step_size
    ) noexcept
    {
        const arma::uword n_groups =
            group_weight.n_rows;
        double *beta_ptr =
            beta.memptr();
        double *eta_ptr =
            eta.memptr();
        const double *gradient_ptr =
            gradient.memptr();
        const double *weight_ptr =
            group_weight.memptr();
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
             * 第一遍：计算梯度步和 L1 soft-threshold 结果。
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
             * 第二遍：更新 beta 和 eta。
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
                beta_ptr[j] =
                    new_beta;
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
     * 平均 Logistic 负对数似然：
     *
     *   1/n * sum_i [
     *       log(1 + exp(eta_i)) - y_i * eta_i
     *   ]
     */
    inline double logistic_nll_mean(
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
        double value = 0.0;
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            const double eta_i =
                eta_ptr[i];
            const double softplus =
                eta_i > 0.0
                ? eta_i + std::log1p(
                    std::exp(-eta_i)
                )
                : std::log1p(
                    std::exp(eta_i)
                );
            value +=
                softplus -
                y_ptr[i] * eta_i;
        }
        return value /
               static_cast<double>(n);
    }


    /*
     * 计算 Logistic SGL 总目标函数。
     */
    inline double logistic_sparse_group_objective(
        const arma::mat& beta,
        const arma::mat& eta,
        const arma::mat& y,
        const GroupLayout& layout,
        const arma::mat& group_weight,
        const double lambda,
        const double alpha
    ) noexcept
    {
        const double loss =
            logistic_nll_mean(
            eta,
            y
            );
        const double *beta_ptr =
            beta.memptr();
        const double *weight_ptr =
            group_weight.memptr();
        double l1_norm = 0.0;
        for (arma::uword j = 0;
        j < beta.n_rows;
        ++j)
    {
        l1_norm +=
            std::abs(beta_ptr[j]);
        }
        double group_norm = 0.0;
        for (arma::uword g = 0;
        g < group_weight.n_rows;
        ++g)
    {
        const arma::uword begin =
            layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            double squared_norm = 0.0;
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                squared_norm +=
                    beta_ptr[j] *
                    beta_ptr[j];
            }
            group_norm +=
                weight_ptr[g] *
                std::sqrt(squared_norm);
        }
        return loss +
               lambda * (
                   alpha * l1_norm +
                   (1.0 - alpha) * group_norm
               );
    }

} // namespace sgl

#endif