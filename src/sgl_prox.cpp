#include <RcppArmadillo.h>
#include <cmath>

#include "sgl_prox.h"

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

    inline void check_penalty_parameter(
        const double value,
              const char *name
    )
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            Rcpp::stop(
                "%s must be a finite non-negative scalar.",
                name
            );
        }
    }

} // anonymous namespace


// [[Rcpp::export]]
arma::mat sgl_sparse_group_prox_cpp(
    arma::mat z,
    const double lambda_l1,
    const double lambda_group,
    const double group_weight = 1.0
)
{
    check_column_matrix(z, "z");
    check_penalty_parameter(
        lambda_l1,
        "lambda_l1"
    );
    check_penalty_parameter(
        lambda_group,
        "lambda_group"
    );
    check_penalty_parameter(
        group_weight,
        "group_weight"
    );
    sgl::sparse_group_prox_inplace(
        z,
        lambda_l1,
        lambda_group,
        group_weight
    );
    return z;
}