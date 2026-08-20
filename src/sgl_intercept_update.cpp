#include <RcppArmadillo.h>
#include <cmath>

#include "sgl_intercept_update.hpp"

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


    inline void check_finite_scalar(
        const double value,
              const char *name
    )
    {
        if (!std::isfinite(value))
        {
            Rcpp::stop(
                "%s must be finite.",
                name
            );
        }
    }


    inline void check_binary_response(
        const arma::mat& y
    )
    {
        const double *y_ptr =
            y.memptr();
        for (arma::uword i = 0;
                i < y.n_rows;
                ++i)
        {
            if (y_ptr[i] != 0.0 &&
                    y_ptr[i] != 1.0)
            {
                Rcpp::stop(
                    "y must contain only 0 and 1."
                );
            }
        }
    }


    inline Rcpp::NumericMatrix copy_column_to_r(
        const arma::mat& x
    )
    {
        Rcpp::NumericMatrix result(
            static_cast<R_xlen_t>(x.n_rows),
            1
        );
        const double *source =
            x.memptr();
        for (arma::uword i = 0;
                i < x.n_rows;
                ++i)
        {
            result(
                static_cast<R_xlen_t>(i),
                0
            ) = source[i];
        }
        return result;
    }

} // anonymous namespace


// [[Rcpp::export]]
double sgl_linear_intercept_gradient_cpp(
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
    return sgl::linear_intercept_gradient(
               eta,
               y
           );
}


// [[Rcpp::export]]
double sgl_logistic_intercept_gradient_cpp(
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
    return sgl::logistic_intercept_gradient(
               eta,
               y
           );
}


// [[Rcpp::export]]
double sgl_logistic_intercept_hessian_cpp(
    const arma::mat& eta
)
{
    check_column_matrix(
        eta,
        "eta"
    );
    return sgl::logistic_intercept_hessian(
               eta
           );
}


// [[Rcpp::export]]
Rcpp::List sgl_update_intercept_eta_cpp(
    arma::mat intercept,
    arma::mat eta,
    const double delta_intercept
)
{
    check_intercept_matrix(
        intercept
    );
    check_column_matrix(
        eta,
        "eta"
    );
    check_finite_scalar(
        delta_intercept,
        "delta_intercept"
    );
    sgl::update_intercept_eta_inplace(
        intercept,
        eta,
        delta_intercept
    );
    return Rcpp::List::create(
               Rcpp::_["intercept"] =
                   copy_column_to_r(intercept),
               Rcpp::_["eta"] =
                   copy_column_to_r(eta)
           );
}