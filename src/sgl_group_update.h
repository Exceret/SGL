#ifndef SGL_GROUP_UPDATE_HPP
#define SGL_GROUP_UPDATE_HPP

#include <RcppArmadillo.h>
#include <cmath>

#include "sgl_group_prox.hpp"

namespace sgl
{

    /*
     * Combined group proximal update and eta update.
     *
     * beta  : p x 1
     * eta   : n x 1
     * X     : n x p
     *
     * For every group:
     *
     *   u_j = soft_threshold(beta_j, lambda_l1)
     *
     *   beta_new_g =
     *       max(
     *           1 -
     *           lambda_group * group_weight_g / ||u_g||_2,
     *           0
     *       ) * u_g
     *
     * Then update:
     *
     *   eta <- eta + X %*% (beta_new - beta_old)
     *
     * beta and eta are modified in place.
     *
     * No p x 1 or n x 1 temporary vector is allocated.
     */
    inline void sparse_group_prox_update_eta_inplace(
    arma::mat& beta,
    arma::mat& eta,
    const arma::mat& X,
    const GroupLayout& layout,
    const arma::mat& group_weight,
    const double lambda_l1,
    const double lambda_group
    ) noexcept
    {
        const arma::uword n_groups =
            group_weight.n_rows;
        double *beta_ptr = beta.memptr();
        double *eta_ptr = eta.memptr();
        const double *weight_ptr =
            group_weight.memptr();
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            const arma::uword begin =
                layout.offsets[g];
            const arma::uword end =
                layout.offsets[g + 1];
            const double group_threshold =
                lambda_group * weight_ptr[g];
            double squared_norm = 0.0;
            /*
             * 第一遍：
             * 使用原始 beta 计算 L1 收缩结果和组范数。
             */
            for (arma::uword k = begin; k < end; ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                const double value =
                    group_soft_threshold(
                        beta_ptr[j],
                        lambda_l1
                    );
                squared_norm += value * value;
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
                        group_threshold / group_norm;
                }
            }
            /*
             * 第二遍：
             * 计算最终 beta、更新 beta，并增量更新 eta。
             */
            for (arma::uword k = begin; k < end; ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                const double old_beta =
                    beta_ptr[j];
                const double l1_value =
                    group_soft_threshold(
                        old_beta,
                        lambda_l1
                    );
                const double new_beta =
                    factor * l1_value;
                const double delta =
                    new_beta - old_beta;
                beta_ptr[j] = new_beta;
                if (delta == 0.0)
                {
                    continue;
                }
                const double *x_ptr =
                    X.colptr(j);
                for (arma::uword i = 0; i < X.n_rows; ++i)
                {
                    eta_ptr[i] += x_ptr[i] * delta;
                }
            }
        }
    }

} // namespace sgl

#endif