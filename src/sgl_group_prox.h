#ifndef SGL_GROUP_PROX_H
#define SGL_GROUP_PROX_H

#include <RcppArmadillo.h>
#include <cmath>

namespace sgl
{

    /*
     * member_order:
     *
     *   按组排列的原始变量下标，0-based
     *
     * offsets:
     *
     *   长度为 n_groups + 1
     *   第 g 组成员位于：
     *
     *       member_order[offsets[g] : offsets[g + 1] - 1]
     *
     *   其中 offsets[0] == 0
     *        offsets[n_groups] == p
     */
    struct GroupLayout
    {
        arma::uvec member_order;
        arma::uvec offsets;
    };


    inline double group_soft_threshold(
        const double value,
        const double threshold
    ) noexcept
    {
        if (value > threshold)
    {
        return value - threshold;
    }
    if (value < -threshold)
    {
        return value + threshold;
    }
    return 0.0;
}


/*
 * 使用预计算的 GroupLayout 执行整组稀疏组近端更新。
 *
 * beta         : p x 1
 * group_weight : G x 1
 * member_order : 按组排列的原始变量下标
 * offsets      : 组边界
 *
 * 每个组内的变量顺序仍然是原始 beta 下标的升序，
 * 因此平方范数的浮点累加顺序与原始扫描实现一致。
 */
inline void sparse_group_prox_groups_inplace(
    arma::mat& beta,
    const GroupLayout& layout,
    const arma::mat& group_weight,
    const double lambda_l1,
    const double lambda_group
) noexcept
    {
        const arma::uword n_groups =
            group_weight.n_rows;
        double *beta_ptr = beta.memptr();
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
             * L1 soft-threshold，并计算组内平方范数。
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
                beta_ptr[j] = value;
                squared_norm += value * value;
            }
            if (squared_norm == 0.0)
            {
                continue;
            }
            const double group_norm =
                std::sqrt(squared_norm);
            if (!(group_norm > group_threshold))
            {
                for (arma::uword k = begin; k < end; ++k)
                {
                    const arma::uword j =
                        layout.member_order[k];
                    beta_ptr[j] = 0.0;
                }
                continue;
            }
            const double factor =
                1.0 - group_threshold / group_norm;
            /*
             * 第二遍：
             * 组级 L2 shrinkage。
             */
            for (arma::uword k = begin; k < end; ++k)
            {
                const arma::uword j =
                    layout.member_order[k];
                beta_ptr[j] *= factor;
            }
        }
    }

} // namespace sgl

#endif