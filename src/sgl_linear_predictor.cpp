#include <RcppArmadillo.h>

#include "sgl_linear_predictor.hpp"

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


    inline void check_intercept_matrix(
        const arma::mat& intercept
    )
    {
        if (intercept.n_rows != 1 ||
                intercept.n_cols != 1)
        {
            Rcpp::stop(
                "intercept must be a 1 x 1 matrix."
            );
        }
        if (!intercept.is_finite())
        {
            Rcpp::stop(
                "intercept must contain only finite values."
            );
        }
    }

} // anonymous namespace


// [[Rcpp::export]]
arma::mat sgl_initialize_linear_predictor_cpp(
    const arma::mat& X,
    const arma::mat& beta,
    const arma::mat& intercept
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
    check_intercept_matrix(
        intercept
    );
    if (X.n_cols != beta.n_rows)
    {
        Rcpp::stop(
            "nrow(beta) must equal ncol(X)."
        );
    }
    arma::mat eta(
        X.n_rows,
        1,
        arma::fill::none
    );
    sgl::initialize_linear_predictor_inplace(
        eta,
        X,
        beta,
        intercept
    );
    return eta;
}


// [[Rcpp::export]]
arma::mat sgl_update_linear_predictor_state_cpp(
    arma::mat eta,
    const arma::mat& X,
    const arma::mat& beta_old,
    const arma::mat& beta_new,
    const arma::mat& intercept_old,
    const arma::mat& intercept_new
)
{
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    check_column_matrix(
        eta,
        "eta"
    );
    check_column_matrix(
        beta_old,
        "beta_old"
    );
    check_column_matrix(
        beta_new,
        "beta_new"
    );
    check_intercept_matrix(
        intercept_old
    );
    check_intercept_matrix(
        intercept_new
    );
    if (X.n_rows != eta.n_rows)
    {
        Rcpp::stop(
            "nrow(eta) must equal nrow(X)."
        );
    }
    if (X.n_cols != beta_old.n_rows)
    {
        Rcpp::stop(
            "nrow(beta_old) must equal ncol(X)."
        );
    }
    if (X.n_cols != beta_new.n_rows)
    {
        Rcpp::stop(
            "nrow(beta_new) must equal ncol(X)."
        );
    }
    if (beta_old.n_rows != beta_new.n_rows)
    {
        Rcpp::stop(
            "beta_old and beta_new must have the same "
            "number of rows."
        );
    }
    sgl::update_linear_predictor_state_inplace(
        eta,
        X,
        beta_old,
        beta_new,
        intercept_old,
        intercept_new
    );
    return eta;
}