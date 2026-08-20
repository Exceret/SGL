#include <RcppArmadillo.h>

#include "sgl_logit_loss.h"

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

    inline void check_binary_response(
        const arma::mat& y
    )
    {
        const double *y_ptr = y.memptr();
        for (arma::uword i = 0; i < y.n_rows; ++i)
        {
            if (y_ptr[i] != 0.0 && y_ptr[i] != 1.0)
            {
                Rcpp::stop(
                    "y must contain only 0 and 1."
                );
            }
        }
    }

} // anonymous namespace


// [[Rcpp::export]]
double sgl_logistic_nll_cpp(
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
    check_binary_response(y);
    return sgl::logistic_negative_log_likelihood(
               eta,
               y
           );
}