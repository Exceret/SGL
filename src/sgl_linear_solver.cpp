#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sgl_group_prox.h"
#include "sgl_linear_gradient.h"
#include "sgl_linear_predictor.h"
#include "sgl_linear_solver.h"
#include "sgl_fista.h"

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


    inline void check_scalar_finite(
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


    inline void check_nonnegative_finite(
        const double value,
        const char *name
    )
    {
        if (!std::isfinite(value) ||
                value < 0.0)
        {
            Rcpp::stop(
                "%s must be finite and non-negative.",
                name
            );
        }
    }


    inline void check_unit_interval(
        const double value,
        const char *name
    )
    {
        if (!std::isfinite(value) ||
                value < 0.0 ||
                value > 1.0)
        {
            Rcpp::stop(
                "%s must be in [0, 1].",
                name
            );
        }
    }


    inline sgl::GroupLayout build_group_layout(
        const arma::mat& group_index
    )
    {
        check_column_matrix(
            group_index,
            "group_index"
        );
        const arma::uword p =
            group_index.n_rows;
        if (p == 0)
        {
            Rcpp::stop(
                "group_index must contain at least one variable."
            );
        }
        const double *group_ptr =
            group_index.memptr();
        arma::uword n_groups = 0;
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const double value =
                group_ptr[j];
            if (!std::isfinite(value) ||
                    value < 1.0 ||
                    std::floor(value) != value ||
                    value > static_cast<double>(p))
            {
                Rcpp::stop(
                    "group_index must contain positive integers."
                );
            }
            const arma::uword group =
                static_cast<arma::uword>(value);
            if (group > n_groups)
            {
                n_groups = group;
            }
        }
        arma::uvec group_sizes(
            n_groups,
            arma::fill::zeros
        );
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const arma::uword group =
                static_cast<arma::uword>(
                    group_ptr[j]
                );
            ++group_sizes[group - 1];
        }
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            if (group_sizes[g] == 0)
            {
                Rcpp::stop(
                    "group_index must use consecutive labels "
                    "1,...,G."
                );
            }
        }
        sgl::GroupLayout layout;
        layout.offsets.set_size(
            n_groups + 1
        );
        layout.offsets[0] = 0;
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            layout.offsets[g + 1] =
                layout.offsets[g] +
                group_sizes[g];
        }
        layout.member_order.set_size(p);
        arma::uvec cursor =
            layout.offsets.subvec(
                0,
                n_groups - 1
            );
        /*
         * 按原始变量下标顺序填入每个组，
         * 保持确定性的浮点累加顺序。
         */
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const arma::uword group =
                static_cast<arma::uword>(
                    group_ptr[j]
                ) - 1;
            layout.member_order[cursor[group]] =
                j;
            ++cursor[group];
        }
        return layout;
    }


    inline Rcpp::NumericMatrix copy_arma_matrix(
        const arma::mat& x
    )
    {
        Rcpp::NumericMatrix result(
            static_cast<R_xlen_t>(x.n_rows),
            static_cast<R_xlen_t>(x.n_cols)
        );
        const double *source =
            x.memptr();
        for (arma::uword j = 0;
                j < x.n_cols;
                ++j)
        {
            for (arma::uword i = 0;
                    i < x.n_rows;
                    ++i)
            {
                result(
                    static_cast<R_xlen_t>(i),
                    static_cast<R_xlen_t>(j)
                ) = source[i + j * x.n_rows];
            }
        }
        return result;
    }


    inline void validate_group_weight(
        const arma::mat& group_weight,
        const arma::uword n_groups
    )
    {
        check_column_matrix(
            group_weight,
            "group_weight"
        );
        if (group_weight.n_rows != n_groups)
        {
            Rcpp::stop(
                "nrow(group_weight) must equal the number "
                "of groups."
            );
        }
        const double *weight_ptr =
            group_weight.memptr();
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            if (!std::isfinite(weight_ptr[g]) ||
                    weight_ptr[g] < 0.0)
            {
                Rcpp::stop(
                    "group_weight must contain finite "
                    "non-negative values."
                );
            }
        }
    }


    inline double max_abs_matrix(
        const arma::mat& x
    )
    {
        double result = 0.0;
        const double *ptr =
            x.memptr();
        for (arma::uword i = 0;
                i < x.n_elem;
                ++i)
        {
            const double value =
                std::abs(ptr[i]);
            if (value > result)
            {
                result = value;
            }
        }
        return result;
    }

} // anonymous namespace


// [[Rcpp::export]]
Rcpp::List sgl_linear_fit_cpp(
    const arma::mat& X,
    const arma::mat& y,
    const arma::mat& group_index,
    const arma::mat& group_weight,
    const arma::mat& initial_beta,
    const arma::mat& initial_intercept,
    const double lambda,
    const double alpha,
    const double step_size,
    const int max_iter,
    const double tol,
    const bool fit_intercept
)
{
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    check_column_matrix(
        y,
        "y"
    );
    check_column_matrix(
        initial_beta,
        "initial_beta"
    );
    if (X.n_rows == 0)
    {
        Rcpp::stop(
            "X must contain at least one observation."
        );
    }
    if (X.n_cols == 0)
    {
        Rcpp::stop(
            "X must contain at least one variable."
        );
    }
    if (X.n_rows != y.n_rows)
    {
        Rcpp::stop(
            "nrow(y) must equal nrow(X)."
        );
    }
    if (X.n_cols != initial_beta.n_rows)
    {
        Rcpp::stop(
            "nrow(initial_beta) must equal ncol(X)."
        );
    }
    if (initial_intercept.n_rows != 1 ||
            initial_intercept.n_cols != 1)
    {
        Rcpp::stop(
            "initial_intercept must be a 1 x 1 matrix."
        );
    }
    if (!initial_intercept.is_finite())
    {
        Rcpp::stop(
            "initial_intercept must contain only "
            "finite values."
        );
    }
    check_nonnegative_finite(
        lambda,
        "lambda"
    );
    check_unit_interval(
        alpha,
        "alpha"
    );
    if (!std::isfinite(step_size) ||
            step_size <= 0.0)
    {
        Rcpp::stop(
            "step_size must be finite and positive."
        );
    }
    if (max_iter <= 0)
    {
        Rcpp::stop(
            "max_iter must be positive."
        );
    }
    if (!std::isfinite(tol) ||
            tol <= 0.0)
    {
        Rcpp::stop(
            "tol must be finite and positive."
        );
    }
    const sgl::GroupLayout layout =
        build_group_layout(group_index);
    validate_group_weight(
        group_weight,
        layout.offsets.n_elem - 1
    );
    arma::mat beta =
        initial_beta;
    arma::mat intercept =
        initial_intercept;
    if (!fit_intercept)
    {
        intercept(0, 0) = 0.0;
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
    arma::mat gradient(
        X.n_cols,
        1,
        arma::fill::none
    );
    arma::mat beta_extrapolated =
        beta;
    arma::mat intercept_extrapolated =
        intercept;
    arma::mat eta_extrapolated =
        eta;
    double fista_t =
        1.0;
    const double lambda_l1 =
        step_size * lambda * alpha;
    const double lambda_group =
        step_size * lambda * (1.0 - alpha);
    bool converged = false;
    int iterations = 0;
    for (int iter = 0;
            iter < max_iter;
            ++iter)
    {
        /*
         * 在 extrapolated eta 上计算梯度。
         */
        sgl::linear_gradient(
            gradient,
            X,
            eta_extrapolated,
            y
        );
        const double inverse_n =
            1.0 /
            static_cast<double>(
                X.n_rows
            );
        double *gradient_ptr =
            gradient.memptr();
        for (arma::uword j = 0;
                j < gradient.n_elem;
                ++j)
        {
            gradient_ptr[j] *=
                inverse_n;
        }
        double intercept_gradient =
            0.0;
        if (fit_intercept)
        {
            const double *eta_ptr =
                eta_extrapolated.memptr();
            const double *y_ptr =
                y.memptr();
            for (arma::uword i = 0;
                    i < eta_extrapolated.n_rows;
                    ++i)
            {
                intercept_gradient +=
                    eta_ptr[i] - y_ptr[i];
            }
            intercept_gradient *=
                inverse_n;
        }
        /*
         * proximal 更新 extrapolated 状态。
         */
        sgl::sparse_group_proximal_gradient_update_eta_inplace(
            beta_extrapolated,
            eta_extrapolated,
            X,
            gradient,
            layout,
            group_weight,
            step_size,
            lambda_l1,
            lambda_group
        );
        if (fit_intercept)
        {
            const double delta_intercept =
                -step_size *
                intercept_gradient;
            intercept_extrapolated(0, 0) +=
                delta_intercept;
            double *eta_extrapolated_ptr =
                eta_extrapolated.memptr();
            for (arma::uword i = 0;
                    i < eta_extrapolated.n_rows;
                    ++i)
            {
                eta_extrapolated_ptr[i] +=
                    delta_intercept;
            }
        }
        const double fista_t_next =
            sgl::fista_next_t(
                fista_t
            );
        const double coefficient =
            (fista_t - 1.0) /
            fista_t_next;
        const double beta_change =
            sgl::fista_commit_and_extrapolate_inplace(
                beta,
                beta_extrapolated,
                eta,
                eta_extrapolated,
                coefficient
            );
        double intercept_change =
            0.0;
        if (fit_intercept)
        {
            intercept_change =
                sgl::fista_commit_scalar_and_extrapolate_inplace(
                    intercept,
                    intercept_extrapolated,
                    coefficient
                );
        }
        const double change =
            std::max(
                beta_change,
                intercept_change
            );
        const double scale =
            1.0 +
            max_abs_matrix(beta) +
            std::abs(
                intercept(0, 0)
            );
        fista_t =
            fista_t_next;
        iterations =
            iter + 1;
        if (change <= tol * scale)
        {
            converged =
                true;
            break;
        }
    }
    const double objective =
        sgl::linear_sparse_group_objective(
            beta,
            eta,
            y,
            layout,
            group_weight,
            lambda,
            alpha
        );
    return Rcpp::List::create(
               Rcpp::_["beta"] =
                   copy_arma_matrix(beta),
               Rcpp::_["intercept"] =
                   copy_arma_matrix(intercept),
               Rcpp::_["eta"] =
                   copy_arma_matrix(eta),
               Rcpp::_["prediction"] =
                   copy_arma_matrix(eta),
               Rcpp::_["objective"] =
                   objective,
               Rcpp::_["iterations"] =
                   iterations,
               Rcpp::_["converged"] =
                   converged
           );
}