#include <RcppArmadillo.h>
#include <cmath>

#include "sgl_linear_algebra.h"

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
            Rcpp::stop("%s must be a single-column matrix.", name);
        }
        if (!x.is_finite())
        {
            Rcpp::stop("%s must contain only finite values.", name);
        }
    }

    inline void check_group_range(
        const arma::mat& X,
              const int first_col,
              const int last_col
    )
    {
        if (first_col < 0 ||
                last_col < 0 ||
                first_col > last_col ||
                static_cast<arma::uword>(last_col) >= X.n_cols)
        {
            Rcpp::stop("Invalid zero-based group column range.");
        }
    }

} // anonymous namespace


// [[Rcpp::export]]
arma::mat sgl_update_eta_cpp(
    const arma::mat& X,
    arma::mat eta,
    const arma::mat& delta_beta
)
{
    if (!X.is_finite())
    {
        Rcpp::stop("X must contain only finite values.");
    }
    check_column_matrix(eta, "eta");
    check_column_matrix(delta_beta, "delta_beta");
    if (X.n_rows != eta.n_rows)
    {
        Rcpp::stop("nrow(eta) must equal nrow(X).");
    }
    if (X.n_cols != delta_beta.n_rows)
    {
        Rcpp::stop("nrow(delta_beta) must equal ncol(X).");
    }
    sgl::update_eta_inplace(
        eta,
        X,
        delta_beta
    );
    return eta;
}


// [[Rcpp::export]]
arma::mat sgl_update_eta_group_cpp(
    const arma::mat& X,
    arma::mat eta,
    const arma::mat& delta_beta,
    const int first_col,
    const int last_col
)
{
    if (!X.is_finite())
    {
        Rcpp::stop("X must contain only finite values.");
    }
    check_column_matrix(eta, "eta");
    check_column_matrix(delta_beta, "delta_beta");
    if (X.n_rows != eta.n_rows)
    {
        Rcpp::stop("nrow(eta) must equal nrow(X).");
    }
    if (X.n_cols != delta_beta.n_rows)
    {
        Rcpp::stop("nrow(delta_beta) must equal ncol(X).");
    }
    check_group_range(X, first_col, last_col);
    sgl::update_eta_group_inplace(
        eta,
        X,
        delta_beta,
        static_cast<arma::uword>(first_col),
        static_cast<arma::uword>(last_col)
    );
    return eta;
}


// [[Rcpp::export]]
arma::mat sgl_gradient_xt_residual_cpp(
    const arma::mat& X,
    const arma::mat& residual
)
{
    if (!X.is_finite())
    {
        Rcpp::stop("X must contain only finite values.");
    }
    check_column_matrix(residual, "residual");
    if (X.n_rows != residual.n_rows)
    {
        Rcpp::stop("nrow(residual) must equal nrow(X).");
    }
    arma::mat gradient(
        X.n_cols,
        1,
        arma::fill::zeros
    );
    sgl::gradient_xt_residual(
        gradient,
        X,
        residual
    );
    return gradient;
}


// [[Rcpp::export]]
arma::mat sgl_gradient_group_xt_residual_cpp(
    const arma::mat& X,
    const arma::mat& residual,
    const int first_col,
    const int last_col
)
{
    if (!X.is_finite())
    {
        Rcpp::stop("X must contain only finite values.");
    }
    check_column_matrix(residual, "residual");
    if (X.n_rows != residual.n_rows)
    {
        Rcpp::stop("nrow(residual) must equal nrow(X).");
    }
    check_group_range(X, first_col, last_col);
    arma::mat gradient(
        X.n_cols,
        1,
        arma::fill::zeros
    );
    sgl::gradient_group_xt_residual(
        gradient,
        X,
        residual,
        static_cast<arma::uword>(first_col),
        static_cast<arma::uword>(last_col)
    );
    return gradient;
}