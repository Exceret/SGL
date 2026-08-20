#include <Rcpp.h>

#include <cmath>

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
Rcpp::List sgl_center_scale_cpp(
    const Rcpp::NumericMatrix& X,
    const bool standardize = true
)
{
    const R_xlen_t n =
        X.nrow();
    const R_xlen_t p =
        X.ncol();
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
    if (!Rcpp::is_true(
                Rcpp::all(Rcpp::is_finite(X))
            ))
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    /*
     * 保持原接口约定：
     *
     *   X.transform: p x 2
     *
     *   第 1 列：列均值
     *   第 2 列：缩放因子
     */
    Rcpp::NumericMatrix X_transform(
        p,
        2
    );
    Rcpp::NumericMatrix X_scaled(
        n,
        p
    );
    for (R_xlen_t j = 0;
            j < p;
            ++j)
    {
        double sum = 0.0;
        for (R_xlen_t i = 0;
                i < n;
                ++i)
        {
            sum += X(i, j);
        }
        const double center =
            sum / static_cast<double>(n);
        double scale =
            1.0;
        if (standardize && n > 1)
        {
            double squared_sum = 0.0;
            for (R_xlen_t i = 0;
                    i < n;
                    ++i)
            {
                const double centered =
                    X(i, j) - center;
                squared_sum +=
                    centered * centered;
            }
            /*
             * 与 R::sd() 一致，使用 n - 1。
             */
            const double variance =
                squared_sum /
                static_cast<double>(n - 1);
            scale = std::sqrt(variance);
            /*
             * 常数列或退化列不进行除零。
             * 这时保持中心化结果，并令 scale = 1。
             */
            if (!std::isfinite(scale) ||
                    scale <= 0.0)
            {
                scale = 1.0;
            }
        }
        X_transform(j, 0) =
            center;
        X_transform(j, 1) =
            scale;
        for (R_xlen_t i = 0;
                i < n;
                ++i)
        {
            X_scaled(i, j) =
                (X(i, j) - center) /
                scale;
        }
    }
    return Rcpp::List::create(
               Rcpp::_["x"] =
                   X_scaled,
               Rcpp::_["X.transform"] =
                   X_transform
           );
}