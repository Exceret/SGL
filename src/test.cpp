#include <RcppArmadillo.h>
#include "sgl_prox.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::vec sgl_prox_test(
    arma::vec z,
    const double lambda_l1,
    const double lambda_group,
    const double group_weight = 1.0
)
{
    if (!std::isfinite(lambda_l1) ||
            !std::isfinite(lambda_group) ||
            !std::isfinite(group_weight))
    {
        Rcpp::stop("Penalty parameters must be finite.");
    }
    if (lambda_l1 < 0.0 ||
            lambda_group < 0.0 ||
            group_weight < 0.0)
    {
        Rcpp::stop("Penalty parameters must be non-negative.");
    }
    sgl::sparse_group_prox_inplace(
        z,
        lambda_l1,
        lambda_group,
        group_weight
    );
    return z;
}