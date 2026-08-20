#include <RcppArmadillo.h>

#include "sgl_group_norm.h"

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

} // anonymous namespace


// [[Rcpp::export]]
double sgl_group_l2_norm_cpp(
    const arma::mat& z
)
{
    check_column_matrix(
        z,
        "z"
    );
    return sgl::group_l2_norm(z);
}