#include <RcppArmadillo.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List sgl_center_scale_cpp(
    const arma::mat& X,
    const bool standardize = true
)
{
    const arma::uword n =
        X.n_rows;
    const arma::uword p =
        X.n_cols;
    if (n == 0)
    {
        Rcpp::stop(
            "X must contain at least one row."
        );
    }
    if (p == 0)
    {
        Rcpp::stop(
            "X must contain at least one column."
        );
    }
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    /*
     * p x 1 的列均值。
     */
    arma::rowvec X_means =
        arma::mean(X, 0);
    /*
     * 中心化矩阵。
     */
    arma::mat X_centered =
        X;
    X_centered.each_row() -=
                  X_means;
    /*
     * 原 SGL 的 scale：
     *
     *   sqrt(sum((X[, j] - mean_j)^2))
     *
     * 注意：这里不除以 n，也不除以 n - 1。
     */
    arma::rowvec X_scale(
        p,
        arma::fill::none
    );
    for (arma::uword j = 0;
            j < p;
            ++j)
    {
        double squared_norm = 0.0;
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            const double value =
                X_centered(i, j);
            squared_norm +=
                value * value;
        }
        X_scale[j] =
            std::sqrt(squared_norm);
        /*
         * 常数列避免除零。
         * X_centered 该列本身为 0。
         */
        if (!std::isfinite(X_scale[j]) ||
                X_scale[j] <= 0.0)
        {
            X_scale[j] = 1.0;
        }
    }
    /*
     * standardize = TRUE：
     *
     *   X_scaled[, j] =
     *       X_centered[, j] / X_scale[j]
     *
     * standardize = FALSE：
     *
     *   只中心化，不缩放；
     *   但 X.transform 第二列仍保存真实 scale。
     */
    arma::mat X_scaled =
        X_centered;
    if (standardize)
    {
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            X_scaled.col(j) /=
                        X_scale[j];
        }
    }
    /*
     * X.transform 为 p x 2 matrix：
     *
     *   第 1 列：X.means
     *   第 2 列：X.scale
     */
    arma::mat X_transform(
        p,
        2,
        arma::fill::none
    );
    for (arma::uword j = 0;
            j < p;
            ++j)
    {
        X_transform(j, 0) =
            X_means[j];
        X_transform(j, 1) =
            X_scale[j];
    }
    return Rcpp::List::create(
               Rcpp::_["x"] =
                   X_scaled,
               Rcpp::_["X.transform"] =
                   X_transform
           );
}