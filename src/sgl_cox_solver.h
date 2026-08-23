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

    struct CoxGradientWorkspace
    {
        std::vector<double> risk_xsum;
        std::vector<double> scaled_weight;

        CoxGradientWorkspace(
            const arma::uword n,
            const arma::uword p
        )
            : risk_xsum(p, 0.0),
              scaled_weight(n, 0.0)
        {
        }
    };
    /*
     * Breslow negative partial-log-likelihood gradient.
     *
     * Inputs:
     *
     *   X       : n x p
     *   eta     : n x 1
     *   status  : n x 1, with values 0/1
     *   layout  : risk-set layout sorted by time in decreasing order
     *
     * Returns:
     *
     *   mean negative partial-log-likelihood
     *
     * gradient:
     *
     *   1 / n_active *
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
        CoxGradientWorkspace& workspace,
        const bool calculate_objective
    ) noexcept
    {
        const arma::uword p =
            X.n_cols;
        // const arma::uword n =
        //     X.n_rows;
        const double *eta_ptr =
            eta.memptr();
        const double *status_ptr =
            status.memptr();
        const arma::uword *order_ptr =
            layout.order.memptr();
        double *gradient_ptr =
            gradient.memptr();
        /*
         * Clear the gradient.
         */
        std::fill(
            gradient_ptr,
            gradient_ptr + p,
            0.0
        );
        std::vector<double> &risk_xsum =
            workspace.risk_xsum;
        std::vector<double> &scaled_weight =
            workspace.scaled_weight;
        std::fill(
            risk_xsum.begin(),
            risk_xsum.end(),
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
             * First compute the maximum eta after adding the current time group.
             *
             * Compared with updating risk_max sample by sample,
             * here each time group performs only a single overall rescale.
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
             * Rescale the previous risk set from risk_max
             * to group_max.
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
             * Compute the scaled weight of each sample in the current time group,
             * while accumulating the risk-set denominator and the sum of event eta.
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
             * Here X is accessed column by column.
             *
             * X.colptr(j) is friendlier to Armadillo's column-major layout,
             * avoiding the previous X(i, j) row-wise access across columns.
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
                     * Add the current time group to the risk set.
                     *
                     * The risk-set statistics and the event-sample statistics
                     * are computed in the same column scan.
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
                 * The current time group has no events, but X must still be
                 * added to the risk set.
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
             * Add the current time group to the risk set.
             *
             * risk_sum is scaled by risk_max to avoid overflow of exp(eta).
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
     * Cox eta initialization:
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
         * Keep the X parameter to avoid changing the existing call interface.
         * The objective function itself no longer needs to access X.
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