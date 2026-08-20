#ifndef SGL_COX_SOLVER_HPP
#define SGL_COX_SOLVER_HPP

#include <RcppArmadillo.h>

#include <cmath>
#include <limits>
#include <vector>

#include "sgl_linear_solver.hpp"

namespace sgl
{

    struct CoxRiskLayout
    {
        arma::uvec order;
        arma::uvec offsets;
        arma::uvec event_counts;
        arma::uword n_events;
    };


    /*
     * Breslow 负部分对数似然梯度。
     *
     * 输入：
     *
     *   X       : n x p
     *   eta     : n x 1
     *   status  : n x 1，取值为 0/1
     *   layout  : 按 time 降序排列的风险集布局
     *
     * 返回：
     *
     *   mean negative partial-log-likelihood
     *
     * gradient：
     *
     *   1 / n_events *
     *   sum_g [
     *       d_g * E_g(X | R_g)
     *       - sum_{i in D_g} X_i
     *   ]
     */
    inline double cox_breslow_gradient_objective_inplace(
        arma::mat& gradient,
        const arma::mat& X,
        const arma::mat& eta,
        const arma::mat& status,
        const CoxRiskLayout& layout,
        const bool calculate_objective
    ) noexcept
    {
        const arma::uword p =
            X.n_cols;
        const double *eta_ptr =
            eta.memptr();
        const double *status_ptr =
            status.memptr();
        double *gradient_ptr =
            gradient.memptr();
        for (arma::uword j = 0; j < p; ++j)
        {
            gradient_ptr[j] = 0.0;
        }
        /*
         * risk_xsum 使用当前最大 eta 进行缩放，
         * 防止 exp(eta) 直接溢出。
         */
        std::vector<double> risk_xsum(
            p,
            0.0
        );
        double risk_sum = 0.0;
        double risk_max =
            -std::numeric_limits<double>::infinity();
        double objective = 0.0;
        for (arma::uword g = 0;
                g + 1 < layout.offsets.n_elem;
                ++g)
        {
            const arma::uword begin =
                layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            /*
             * 将当前时间组加入风险集。
             * 因为 order 按 time 降序排列，
             * 当前累积集合即为 R(t_g)。
             */
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword i =
                    layout.order[k];
                const double eta_i =
                    eta_ptr[i];
                if (risk_sum == 0.0)
                {
                    risk_max = eta_i;
                    risk_sum = 1.0;
                    for (arma::uword j = 0;
                            j < p;
                            ++j)
                    {
                        risk_xsum[j] =
                            X(i, j);
                    }
                    continue;
                }
                if (eta_i <= risk_max)
                {
                    const double scaled_weight =
                        std::exp(eta_i - risk_max);
                    risk_sum += scaled_weight;
                    for (arma::uword j = 0;
                            j < p;
                            ++j)
                    {
                        risk_xsum[j] +=
                            scaled_weight * X(i, j);
                    }
                }
                else
                {
                    const double rescale =
                        std::exp(risk_max - eta_i);
                    risk_sum *= rescale;
                    for (arma::uword j = 0;
                            j < p;
                            ++j)
                    {
                        risk_xsum[j] *= rescale;
                        risk_xsum[j] += X(i, j);
                    }
                    risk_max = eta_i;
                }
            }
            const arma::uword n_events_group =
                layout.event_counts[g];
            if (n_events_group == 0)
            {
                continue;
            }
            const double event_count =
                static_cast<double>(
                    n_events_group
                );
            /*
             * 计算：
             *
             *   d_g * risk_mean - event_sum
             */
            for (arma::uword j = 0;
                    j < p;
                    ++j)
            {
                double event_sum =
                    0.0;
                for (arma::uword k = begin;
                        k < end;
                        ++k)
                {
                    const arma::uword i =
                        layout.order[k];
                    if (status_ptr[i] == 1.0)
                    {
                        event_sum +=
                            X(i, j);
                    }
                }
                const double risk_mean =
                    risk_xsum[j] / risk_sum;
                gradient_ptr[j] +=
                    event_count * risk_mean -
                    event_sum;
            }
            if (calculate_objective)
            {
                double event_eta_sum =
                    0.0;
                for (arma::uword k = begin;
                        k < end;
                        ++k)
                {
                    const arma::uword i =
                        layout.order[k];
                    if (status_ptr[i] == 1.0)
                    {
                        event_eta_sum +=
                            eta_ptr[i];
                    }
                }
                const double log_risk_sum =
                    risk_max +
                    std::log(risk_sum);
                objective +=
                    event_count * log_risk_sum -
                    event_eta_sum;
            }
        }
        const double inverse_events =
            1.0 /
            static_cast<double>(
                layout.n_events
            );
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            gradient_ptr[j] *=
                inverse_events;
        }
        if (!calculate_objective)
        {
            return 0.0;
        }
        return objective *
               inverse_events;
    }


    /*
     * Cox eta 初始化：
     *
     *   eta = X %*% beta
     */
    inline void initialize_cox_eta_inplace(
        arma::mat& eta,
        const arma::mat& X,
        const arma::mat& beta
    ) noexcept
    {
        const arma::uword n =
            X.n_rows;
        const arma::uword p =
            X.n_cols;
        double *eta_ptr =
            eta.memptr();
        const double *beta_ptr =
            beta.memptr();
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            eta_ptr[i] = 0.0;
        }
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const double beta_j =
                beta_ptr[j];
            if (beta_j == 0.0)
            {
                continue;
            }
            const double *x_ptr =
                X.colptr(j);
            for (arma::uword i = 0;
                    i < n;
                    ++i)
            {
                eta_ptr[i] +=
                    x_ptr[i] * beta_j;
            }
        }
    }


    inline double cox_sparse_group_objective(
        const arma::mat& beta,
        const arma::mat& X,
        const arma::mat& eta,
        const arma::mat& status,
        const CoxRiskLayout& risk_layout,
        const arma::mat& group_weight,
        const sgl::GroupLayout& group_layout,
        const double lambda,
        const double alpha
    ) noexcept
    {
        arma::mat gradient(
        beta.n_rows,
        1,
        arma::fill::none
        );
        const double loss =
            cox_breslow_gradient_objective_inplace(
            gradient,
            X,
            eta,
            status,
            risk_layout,
            true
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
            group_layout.offsets[g];
            const arma::uword end =
                group_layout.offsets[g + 1];
            double squared_norm = 0.0;
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword j =
                    group_layout.member_order[k];
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