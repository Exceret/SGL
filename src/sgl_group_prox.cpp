#include <RcppArmadillo.h>
#include <cmath>
#include <vector>

#include "sgl_group_prox.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

namespace
{

    inline void check_column_matrix(
    const arma::mat& x,
    const char *name
    )
    {
        if (x.n_cols != 1)
        {
            Rcpp::stop(
                "%s must be a single-column matrix.",
                name
            );
        }
        if (!x.is_finite())
        {
            Rcpp::stop(
                "%s must contain only finite values.",
                name
            );
        }
    }


    inline void check_nonnegative_finite(
        const double value,
              const char *name
    )
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            Rcpp::stop(
                "%s must be finite and non-negative.",
                name
            );
        }
    }


    inline arma::uword checked_integer_index(
        const double value,
        const arma::uword upper_bound,
        const char *name
    )
    {
        if (!std::isfinite(value) ||
                value < 0.0 ||
                std::floor(value) != value ||
                value >= static_cast<double>(upper_bound))
        {
            Rcpp::stop(
                "%s must contain valid zero-based integer indices.",
                name
            );
        }
        return static_cast<arma::uword>(value);
    }


    inline arma::uword checked_group_label(
        const double value,
        const arma::uword p
    )
    {
        if (!std::isfinite(value) ||
                value < 1.0 ||
                std::floor(value) != value ||
                value > static_cast<double>(p))
        {
            Rcpp::stop(
                "group_index must contain positive integers."
            );
        }
        return static_cast<arma::uword>(value);
    }


    /*
     * 根据 p x 1 的 group_index 构造预计算布局。
     *
     * group_index 使用 1-based 连续组标签：
     *
     *   1, 2, ..., G
     */
    inline sgl::GroupLayout build_group_layout(
        const arma::mat& group_index
    )
    {
        const arma::uword p =
            group_index.n_rows;
        if (p == 0)
        {
            Rcpp::stop(
                "group_index must contain at least one variable."
            );
        }
        const double *group_ptr =
            group_index.memptr();
        arma::uword n_groups = 0;
        for (arma::uword j = 0; j < p; ++j)
        {
            const arma::uword group =
                checked_group_label(
                    group_ptr[j],
                    p
                );
            if (group > n_groups)
            {
                n_groups = group;
            }
        }
        arma::uvec group_sizes(
            n_groups,
            arma::fill::zeros
        );
        for (arma::uword j = 0; j < p; ++j)
        {
            const arma::uword group =
                static_cast<arma::uword>(
                    group_ptr[j]
                );
            ++group_sizes[group - 1];
        }
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            if (group_sizes[g] == 0)
            {
                Rcpp::stop(
                    "group_index must use consecutive labels "
                    "1,...,G."
                );
            }
        }
        sgl::GroupLayout layout;
        layout.offsets.set_size(n_groups + 1);
        layout.offsets[0] = 0;
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            layout.offsets[g + 1] =
                layout.offsets[g] +
                group_sizes[g];
        }
        layout.member_order.set_size(p);
        /*
         * cursor 保证每组成员按原始变量下标升序排列。
         */
        arma::uvec cursor =
            layout.offsets.subvec(0, n_groups - 1);
        for (arma::uword j = 0; j < p; ++j)
        {
            const arma::uword group =
                static_cast<arma::uword>(
                    group_ptr[j]
                ) - 1;
            layout.member_order[cursor[group]] = j;
            ++cursor[group];
        }
        return layout;
    }


    inline Rcpp::NumericMatrix uvec_to_numeric_column_matrix(
        const arma::uvec& x
    )
    {
        Rcpp::NumericMatrix result(
            static_cast<R_xlen_t>(x.n_elem),
            1
        );
        for (arma::uword i = 0; i < x.n_elem; ++i)
        {
            result(
                static_cast<R_xlen_t>(i),
                0
            ) = static_cast<double>(x[i]);
        }
        return result;
    }


    inline sgl::GroupLayout validate_cached_layout(
        const arma::mat& member_order,
        const arma::mat& offsets,
        const arma::mat& group_weight,
        const arma::uword p
    )
    {
        check_column_matrix(
            member_order,
            "member_order"
        );
        check_column_matrix(
            offsets,
            "offsets"
        );
        check_column_matrix(
            group_weight,
            "group_weight"
        );
        if (offsets.n_rows < 2)
        {
            Rcpp::stop(
                "offsets must contain at least two values."
            );
        }
        const arma::uword n_groups =
            offsets.n_rows - 1;
        if (group_weight.n_rows != n_groups)
        {
            Rcpp::stop(
                "nrow(group_weight) must equal "
                "nrow(offsets) - 1."
            );
        }
        if (member_order.n_rows != p)
        {
            Rcpp::stop(
                "nrow(member_order) must equal nrow(beta)."
            );
        }
        sgl::GroupLayout layout;
        layout.member_order.set_size(p);
        layout.offsets.set_size(n_groups + 1);
        for (arma::uword i = 0; i < p; ++i)
        {
            layout.member_order[i] =
                checked_integer_index(
                    member_order(i, 0),
                    p,
                    "member_order"
                );
        }
        for (arma::uword i = 0; i <= n_groups; ++i)
        {
            const double value =
                offsets(i, 0);
            if (!std::isfinite(value) ||
                    value < 0.0 ||
                    std::floor(value) != value ||
                    value > static_cast<double>(p))
            {
                Rcpp::stop(
                    "offsets must contain valid integer "
                    "positions."
                );
            }
            layout.offsets[i] =
                static_cast<arma::uword>(value);
        }
        if (layout.offsets[0] != 0 ||
                layout.offsets[n_groups] != p)
        {
            Rcpp::stop(
                "offsets must start at 0 and end at nrow(beta)."
            );
        }
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            if (layout.offsets[g + 1] <= layout.offsets[g])
            {
                Rcpp::stop(
                    "Each group must contain at least one variable."
                );
            }
        }
        std::vector<unsigned char> seen(
            p,
            static_cast<unsigned char>(0)
        );
        for (arma::uword i = 0; i < p; ++i)
        {
            const arma::uword member =
                layout.member_order[i];
            if (seen[member] != 0)
            {
                Rcpp::stop(
                    "member_order must contain each variable "
                    "exactly once."
                );
            }
            seen[member] = 1;
        }
        for (arma::uword i = 0; i < p; ++i)
        {
            if (seen[i] == 0)
            {
                Rcpp::stop(
                    "member_order must contain each variable "
                    "exactly once."
                );
            }
        }
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            check_nonnegative_finite(
                group_weight(g, 0),
                "group_weight"
            );
        }
        return layout;
    }

} // anonymous namespace


// [[Rcpp::export]]
Rcpp::List sgl_group_layout_cpp(
    const arma::mat& group_index
)
{
    check_column_matrix(
        group_index,
        "group_index"
    );
    const sgl::GroupLayout layout =
        build_group_layout(group_index);
    const arma::uword p =
        layout.member_order.n_elem;
    const arma::uword n_groups =
        layout.offsets.n_elem - 1;
    /*
     * 使用 Rcpp::NumericMatrix 直接分配 R 所有的内存。
     * 不通过局部 arma::mat 返回布局元数据。
     */
    Rcpp::NumericMatrix member_order(
        static_cast<R_xlen_t>(p),
        1
    );
    Rcpp::NumericMatrix offsets(
        static_cast<R_xlen_t>(n_groups + 1),
        1
    );
    Rcpp::NumericMatrix group_sizes(
        static_cast<R_xlen_t>(n_groups),
        1
    );
    for (arma::uword i = 0; i < p; ++i)
    {
        member_order(
            static_cast<R_xlen_t>(i),
            0
        ) = static_cast<double>(
                layout.member_order[i]
            );
    }
    for (arma::uword i = 0; i <= n_groups; ++i)
    {
        offsets(
            static_cast<R_xlen_t>(i),
            0
        ) = static_cast<double>(
                layout.offsets[i]
            );
    }
    for (arma::uword g = 0; g < n_groups; ++g)
    {
        group_sizes(
            static_cast<R_xlen_t>(g),
            0
        ) = static_cast<double>(
                layout.offsets[g + 1] -
                layout.offsets[g]
            );
    }
    /*
     * 返回前再次显式 clone，确保列表持有独立的 R 对象。
     */
    return Rcpp::List::create(
               Rcpp::_["member_order"] =
                   Rcpp::clone(member_order),
               Rcpp::_["offsets"] =
                   Rcpp::clone(offsets),
               Rcpp::_["group_sizes"] =
                   Rcpp::clone(group_sizes)
           );
}


// [[Rcpp::export]]
arma::mat sgl_sparse_group_prox_groups_cpp(
    arma::mat beta,
    const arma::mat& group_index,
    const arma::mat& group_weight,
    const double lambda_l1,
    const double lambda_group
)
{
    check_column_matrix(
        beta,
        "beta"
    );
    check_column_matrix(
        group_index,
        "group_index"
    );
    check_column_matrix(
        group_weight,
        "group_weight"
    );
    if (beta.n_rows != group_index.n_rows)
    {
        Rcpp::stop(
            "nrow(group_index) must equal nrow(beta)."
        );
    }
    check_nonnegative_finite(
        lambda_l1,
        "lambda_l1"
    );
    check_nonnegative_finite(
        lambda_group,
        "lambda_group"
    );
    sgl::GroupLayout layout =
        build_group_layout(group_index);
    if (group_weight.n_rows !=
            layout.offsets.n_elem - 1)
    {
        Rcpp::stop(
            "nrow(group_weight) must equal the number "
            "of groups."
        );
    }
    for (arma::uword g = 0;
            g < group_weight.n_rows;
            ++g)
    {
        check_nonnegative_finite(
            group_weight(g, 0),
            "group_weight"
        );
    }
    sgl::sparse_group_prox_groups_inplace(
        beta,
        layout,
        group_weight,
        lambda_l1,
        lambda_group
    );
    return beta;
}


// [[Rcpp::export]]
arma::mat sgl_sparse_group_prox_groups_cached_cpp(
    arma::mat beta,
    const arma::mat& member_order,
    const arma::mat& offsets,
    const arma::mat& group_weight,
    const double lambda_l1,
    const double lambda_group
)
{
    check_column_matrix(
        beta,
        "beta"
    );
    check_nonnegative_finite(
        lambda_l1,
        "lambda_l1"
    );
    check_nonnegative_finite(
        lambda_group,
        "lambda_group"
    );
    sgl::GroupLayout layout =
        validate_cached_layout(
            member_order,
            offsets,
            group_weight,
            beta.n_rows
        );
    sgl::sparse_group_prox_groups_inplace(
        beta,
        layout,
        group_weight,
        lambda_l1,
        lambda_group
    );
    return beta;
}