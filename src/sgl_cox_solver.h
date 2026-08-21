#ifndef SGL_COX_SOLVER_H
#define SGL_COX_SOLVER_H

#include <RcppArmadillo.h>

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

#include "sgl_linear_solver.h"

namespace sgl
{

    struct CoxRiskLayout
    {
        arma::uvec order;
        arma::uvec offsets;
        arma::uvec event_counts;

        arma::uword n_events;
        arma::uword n_active;
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
        const arma::uword n =
            X.n_rows;
        const double *eta_ptr =
            eta.memptr();
        const double *status_ptr =
            status.memptr();
        const arma::uword *order_ptr =
            layout.order.memptr();
        double *gradient_ptr =
            gradient.memptr();
        /*
         * 梯度清零。
         */
        std::fill(
            gradient_ptr,
            gradient_ptr + p,
            0.0
        );
        /*
         * risk_xsum[j] =
         *   sum_{i in R_g} exp(eta_i - risk_max) * X(i, j)
         */
        std::vector<double> risk_xsum(
            p,
            0.0
        );
        /*
         * scaled_weight[i] 保存当前时间组中样本的：
         *
         *   exp(eta_i - group_max)
         *
         * 这样后面按列访问 X 时无需重复计算 exp。
         */
        std::vector<double> scaled_weight(
            n,
            0.0
        );
        double risk_sum =
            0.0;
        double risk_max =
            -std::numeric_limits<double>::infinity();
        double objective =
            0.0;
        for (arma::uword g = 0;
                g + 1 < layout.offsets.n_elem;
                ++g)
        {
            const arma::uword begin =
                layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            /*
             * 先计算加入当前时间组后的最大 eta。
             *
             * 与逐个样本更新 risk_max 相比，
             * 这里每个时间组只进行一次整体 rescale。
             */
            double group_max =
                risk_max;
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword i =
                    order_ptr[k];
                group_max =
                    std::max(
                        group_max,
                        eta_ptr[i]
                    );
            }
            /*
             * 将之前的风险集从 risk_max
             * 缩放到 group_max。
             */
            if (risk_sum != 0.0)
            {
                const double rescale =
                    std::exp(
                        risk_max - group_max
                    );
                risk_sum *=
                    rescale;
                for (arma::uword j = 0;
                        j < p;
                        ++j)
                {
                    risk_xsum[j] *=
                        rescale;
                }
            }
            /*
             * 计算当前时间组中每个样本的缩放权重，
             * 同时累计风险集分母和事件 eta 总和。
             */
            double added_risk_sum =
                0.0;
            double event_eta_sum =
                0.0;
            const arma::uword n_events_group =
                layout.event_counts[g];
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword i =
                    order_ptr[k];
                const double weight =
                    std::exp(
                        eta_ptr[i] - group_max
                    );
                scaled_weight[i] =
                    weight;
                added_risk_sum +=
                    weight;
                if (status_ptr[i] == 1.0)
                {
                    event_eta_sum +=
                        eta_ptr[i];
                }
            }
            risk_sum +=
                added_risk_sum;
            risk_max =
                group_max;
            /*
             * 这里改为按照特征列访问 X。
             *
             * X.colptr(j) 对 Armadillo 的列主序布局更友好，
             * 避免原来的 X(i, j) 行方向跨列访问。
             */
            if (n_events_group != 0)
            {
                const double event_count =
                    static_cast<double>(
                        n_events_group
                    );
                for (arma::uword j = 0;
                        j < p;
                        ++j)
                {
                    const double *x_ptr =
                        X.colptr(j);
                    double current_risk_xsum =
                        risk_xsum[j];
                    double event_sum =
                        0.0;
                    /*
                     * 当前时间组加入风险集。
                     *
                     * 风险集统计和事件样本统计
                     * 在同一次列扫描中完成。
                     */
                    for (arma::uword k = begin;
                            k < end;
                            ++k)
                    {
                        const arma::uword i =
                            order_ptr[k];
                        const double x_value =
                            x_ptr[i];
                        current_risk_xsum +=
                            scaled_weight[i] *
                            x_value;
                        if (status_ptr[i] == 1.0)
                        {
                            event_sum +=
                                x_value;
                        }
                    }
                    risk_xsum[j] =
                        current_risk_xsum;
                    const double risk_mean =
                        current_risk_xsum /
                        risk_sum;
                    gradient_ptr[j] +=
                        event_count *
                        risk_mean -
                        event_sum;
                }
            }
            else
            {
                /*
                 * 当前时间组没有事件，但仍然必须把
                 * X 加入风险集。
                 */
                for (arma::uword j = 0;
                        j < p;
                        ++j)
                {
                    const double *x_ptr =
                        X.colptr(j);
                    double current_risk_xsum =
                        risk_xsum[j];
                    for (arma::uword k = begin;
                            k < end;
                            ++k)
                    {
                        const arma::uword i =
                            order_ptr[k];
                        current_risk_xsum +=
                            scaled_weight[i] *
                            x_ptr[i];
                    }
                    risk_xsum[j] =
                        current_risk_xsum;
                }
            }
            if (calculate_objective)
            {
                const double log_risk_sum =
                    risk_max +
                    std::log(risk_sum);
                objective +=
                    static_cast<double>(
                        n_events_group
                    ) *
                    log_risk_sum -
                    event_eta_sum;
            }
        }
        const double inverse_active = 1.0 / static_cast<double>(layout.n_active);
        for (arma::uword j = 0; j < p; ++j)
        {
            gradient_ptr[j] *=
                inverse_active;
        }
        if (!calculate_objective)
        {
            return 0.0;
        }
        return objective * inverse_active;
    }


    inline double cox_breslow_objective_inplace(
        const arma::mat& eta,
        const arma::mat& status,
        const CoxRiskLayout& layout
    ) noexcept
    {
        const double *eta_ptr =
            eta.memptr();
        const double *status_ptr =
            status.memptr();
        const arma::uword *order_ptr =
            layout.order.memptr();
        double risk_sum =
            0.0;
        double risk_max =
            -std::numeric_limits<double>::infinity();
        double objective =
            0.0;
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
             *
             * risk_sum 使用 risk_max 缩放，
             * 避免 exp(eta) 溢出。
             */
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword i =
                    order_ptr[k];
                const double eta_i =
                    eta_ptr[i];
                if (risk_sum == 0.0)
                {
                    risk_max =
                        eta_i;
                    risk_sum =
                        1.0;
                    continue;
                }
                if (eta_i <= risk_max)
                {
                    risk_sum +=
                        std::exp(
                            eta_i - risk_max
                        );
                }
                else
                {
                    const double rescale =
                        std::exp(
                            risk_max - eta_i
                        );
                    risk_sum *=
                        rescale;
                    risk_sum +=
                        1.0;
                    risk_max =
                        eta_i;
                }
            }
            const arma::uword n_events_group =
                layout.event_counts[g];
            if (n_events_group == 0)
            {
                continue;
            }
            double event_eta_sum =
                0.0;
            for (arma::uword k = begin;
                    k < end;
                    ++k)
            {
                const arma::uword i =
                    order_ptr[k];
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
                static_cast<double>(
                    n_events_group
                ) *
                log_risk_sum -
                event_eta_sum;
        }
        return objective /
               static_cast<double>(
                   layout.n_active
               );
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
        /*
         * 保留 X 参数以避免修改现有调用接口。
         * 目标函数本身不再需要访问 X。
         */
        (void) X;
        const double loss =
            cox_breslow_objective_inplace(
            eta,
            status,
            risk_layout
            );
        const double *beta_ptr =
            beta.memptr();
        const double *weight_ptr =
            group_weight.memptr();
        double l1_norm =
            0.0;
        for (arma::uword j = 0;
        j < beta.n_rows;
        ++j)
    {
        l1_norm +=
            std::abs(
                beta_ptr[j]
            );
        }
        double group_norm =
            0.0;
        for (arma::uword g = 0;
        g < group_weight.n_rows;
        ++g)
    {
        const arma::uword begin =
            group_layout.offsets[g];
            const arma::uword end =
                group_layout.offsets[g + 1];
            double squared_norm =
                0.0;
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
                std::sqrt(
                    squared_norm
                );
        }
        return loss +
               lambda *
               (
                   alpha * l1_norm +
                   (1.0 - alpha) * group_norm
               );
    }

} // namespace sgl

#endif