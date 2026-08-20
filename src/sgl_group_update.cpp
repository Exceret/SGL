#include <RcppArmadillo.h>

#include <cmath>
#include <vector>

#include "sgl_group_prox.h"
#include "sgl_group_update.h"

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
        if (member_order.n_rows != p)
        {
            Rcpp::stop(
                "nrow(member_order) must equal ncol(X)."
            );
        }
        if (offsets.n_rows < 2)
        {
            Rcpp::stop(
                "offsets must contain at least two rows."
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
        sgl::GroupLayout layout;
        layout.member_order.set_size(p);
        layout.offsets.set_size(n_groups + 1);
        const double *member_ptr =
            member_order.memptr();
        const double *offset_ptr =
            offsets.memptr();
        for (arma::uword i = 0; i < p; ++i)
        {
            const double value =
                member_ptr[i];
            if (!std::isfinite(value) ||
                    value < 0.0 ||
                    std::floor(value) != value ||
                    value >= static_cast<double>(p))
            {
                Rcpp::stop(
                    "member_order must contain valid "
                    "zero-based indices."
                );
            }
            layout.member_order[i] =
                static_cast<arma::uword>(value);
        }
        for (arma::uword i = 0;
                i <= n_groups;
                ++i)
        {
            const double value =
                offset_ptr[i];
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
                "offsets must start at 0 and end at ncol(X)."
            );
        }
        for (arma::uword g = 0; g < n_groups; ++g)
        {
            if (layout.offsets[g + 1] <= layout.offsets[g])
            {
                Rcpp::stop(
                    "Each group must contain at least one "
                    "variable."
                );
            }
            check_nonnegative_finite(
                group_weight(g, 0),
                "group_weight"
            );
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
        return layout;
    }


    inline Rcpp::NumericMatrix copy_matrix_to_r(
        const arma::mat& x
    )
    {
        Rcpp::NumericMatrix result(
            static_cast<R_xlen_t>(x.n_rows),
            static_cast<R_xlen_t>(x.n_cols)
        );
        for (arma::uword j = 0; j < x.n_cols; ++j)
        {
            for (arma::uword i = 0; i < x.n_rows; ++i)
            {
                result(
                    static_cast<R_xlen_t>(i),
                    static_cast<R_xlen_t>(j)
                ) = x(i, j);
            }
        }
        return result;
    }

} // anonymous namespace


// [[Rcpp::export]]
Rcpp::List sgl_sparse_group_prox_update_eta_cached_cpp(
    arma::mat beta,
    arma::mat eta,
    const arma::mat& X,
    const arma::mat& member_order,
    const arma::mat& offsets,
    const arma::mat& group_weight,
    const double lambda_l1,
    const double lambda_group
)
{
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    check_column_matrix(
        beta,
        "beta"
    );
    check_column_matrix(
        eta,
        "eta"
    );
    if (X.n_rows != eta.n_rows)
    {
        Rcpp::stop(
            "nrow(eta) must equal nrow(X)."
        );
    }
    if (X.n_cols != beta.n_rows)
    {
        Rcpp::stop(
            "nrow(beta) must equal ncol(X)."
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
    const sgl::GroupLayout layout =
        validate_cached_layout(
            member_order,
            offsets,
            group_weight,
            X.n_cols
        );
    sgl::sparse_group_prox_update_eta_inplace(
        beta,
        eta,
        X,
        layout,
        group_weight,
        lambda_l1,
        lambda_group
    );
    return Rcpp::List::create(
               Rcpp::_["beta"] =
                   copy_matrix_to_r(beta),
               Rcpp::_["eta"] =
                   copy_matrix_to_r(eta)
           );
}