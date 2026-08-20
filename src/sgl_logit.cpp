#include <RcppArmadillo.h>

#include "sgl_logit.h"

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
arma::mat sgl_logistic_probability_cpp(
    const arma::mat& eta
)
{
    check_column_matrix(
        eta,
        "eta"
    );
    arma::mat probability(
        eta.n_rows,
        1,
        arma::fill::none
    );
    sgl::logistic_probability(
        probability,
        eta
    );
    return probability;
}