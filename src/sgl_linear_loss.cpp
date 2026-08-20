#include <RcppArmadillo.h>

#include "sgl_linear_loss.hpp"

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
double sgl_linear_squared_loss_cpp(
    const arma::mat& eta,
    const arma::mat& y
)
{
    check_column_matrix(
        eta,
        "eta"
    );
    check_column_matrix(
        y,
        "y"
    );
    if (eta.n_rows != y.n_rows)
    {
        Rcpp::stop(
            "nrow(y) must equal nrow(eta)."
        );
    }
    return sgl::linear_squared_loss(
               eta,
               y
           );
}