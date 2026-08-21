#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double cvSGL_cox_logsumexp(NumericVector x)
{
    if (x.size() == 0)
    {
        return R_NegInf;
    }
    double maximum = Rcpp::max(x);
    double total = 0.0;
    for (int i = 0; i < x.size(); ++i)
    {
        total += std::exp(x[i] - maximum);
    }
    return maximum + std::log(total);
}

// [[Rcpp::export]]
arma::mat cvSGL_eta_cpp(
    Rcpp::List fit,
    arma::mat X,
    std::string type = "cox"
)
{
    // X.transform
    Rcpp::List X_transform = fit["X.transform"];
    // X.means
    arma::vec X_means =
        Rcpp::as<arma::vec>(X_transform["X.means"]);
    // X.scale
    SEXP scale_sexp = X_transform["X.scale"];
    arma::vec X_scale;
    if (Rf_isNull(scale_sexp))
    {
        X_scale = arma::ones<arma::vec>(X.n_cols);
    }
    else
    {
        X_scale = Rcpp::as<arma::vec>(scale_sexp);
        // 对应 R 中的 rep(X_scale, length.out = ncol(X))
        if (X_scale.n_elem == 0)
        {
            X_scale = arma::ones<arma::vec>(X.n_cols);
        }
        else if (X_scale.n_elem != X.n_cols)
        {
            arma::vec tmp(X.n_cols);
            for (arma::uword j = 0; j < X.n_cols; ++j)
            {
                tmp[j] = X_scale[j % X_scale.n_elem];
            }
            X_scale = tmp;
        }
    }
    // 对应：
    // X_new <- sweep(X, 2, X_means, "-")
    // X_new <- sweep(X_new, 2, X_scale, "/")
    if (X_means.n_elem != X.n_cols)
    {
        Rcpp::stop("length of X.means must equal ncol(X)");
    }
    if (X_scale.n_elem != X.n_cols)
    {
        Rcpp::stop("length of X.scale must equal ncol(X)");
    }
    arma::mat X_new = X;
    for (arma::uword j = 0; j < X.n_cols; ++j)
    {
        X_new.col(j) -= X_means[j];
        X_new.col(j) /= X_scale[j];
    }
    // eta <- X_new %*% fit$beta
    arma::mat beta = Rcpp::as<arma::mat>(fit["beta"]);
    arma::mat eta = X_new * beta;
    // 原代码中的 type 未定义，这里显式使用函数参数 type
    if (type != "cox")
    {
        Rcpp::NumericVector intercept_r = fit["intercept"];
        arma::vec intercept = Rcpp::as<arma::vec>(intercept_r);
        if (intercept.n_elem == 0)
        {
            Rcpp::stop("intercept cannot be empty");
        }
        // 对应 sweep(eta, 2, intercept, "+")
        // R 中 sweep 会按列循环使用 intercept
        for (arma::uword j = 0; j < eta.n_cols; ++j)
        {
            eta.col(j) += intercept[j % intercept.n_elem];
        }
    }
    return eta;
}