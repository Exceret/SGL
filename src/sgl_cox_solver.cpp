#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sgl_cox_solver.h"
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


    inline void check_positive_scalar(
        const double value,
        const char *name
    )
    {
        if (!std::isfinite(value) ||
                value <= 0.0)
        {
            Rcpp::stop(
                "%s must be finite and positive.",
                name
            );
        }
    }


    inline void check_binary_status(
        const arma::mat& status
    )
    {
        const double *status_ptr =
            status.memptr();
        for (arma::uword i = 0;
                i < status.n_rows;
                ++i)
        {
            if (status_ptr[i] != 0.0 &&
                    status_ptr[i] != 1.0)
            {
                Rcpp::stop(
                    "status must contain only 0 and 1."
                );
            }
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
            n_groups = std::max(
                           n_groups,
                           static_cast<arma::uword>(value)
                       );
        }
        arma::uvec group_sizes(
            n_groups,
            arma::fill::zeros
        );
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const arma::uword g =
                static_cast<arma::uword>(
                    group_ptr[j]
                );
            ++group_sizes[g - 1];
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
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const arma::uword g =
                static_cast<arma::uword>(
                    group_ptr[j]
                ) - 1;
            layout.member_order[cursor[g]] = j;
            ++cursor[g];
        }
        return layout;
    }


    inline sgl::CoxRiskLayout build_cox_risk_layout(
        const arma::mat& time,
        const arma::mat& status
    )
    {
        const arma::uword n =
            time.n_rows;
        const double *time_ptr =
            time.memptr();
        const double *status_ptr =
            status.memptr();
        std::vector<arma::uword> order_vec(n);
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            order_vec[i] = i;
        }
        std::stable_sort(
            order_vec.begin(),
            order_vec.end(),
            [&time_ptr](
                const arma::uword a,
                const arma::uword b
            )
        {
            return time_ptr[a] >
                   time_ptr[b];
        }
        );
        std::vector<arma::uword> offsets_vec;
        std::vector<arma::uword> event_counts_vec;
        offsets_vec.push_back(0);
        arma::uword n_events = 0;
        arma::uword group_begin = 0;
        while (group_begin < n)
        {
            const double group_time =
                time_ptr[
                    order_vec[group_begin]
                ];
            arma::uword group_end =
                group_begin + 1;
            while (
                group_end < n &&
                time_ptr[
                    order_vec[group_end]
                ] == group_time
            )
            {
                ++group_end;
            }
            arma::uword group_events = 0;
            for (arma::uword k = group_begin;
                    k < group_end;
                    ++k)
            {
                const arma::uword i =
                    order_vec[k];
                if (status_ptr[i] == 1.0)
                {
                    ++group_events;
                    ++n_events;
                }
            }
            offsets_vec.push_back(group_end);
            event_counts_vec.push_back(group_events);
            group_begin = group_end;
        }
        if (n_events == 0)
        {
            Rcpp::stop(
                "status must contain at least one event."
            );
        }
        sgl::CoxRiskLayout layout;
        layout.order.set_size(n);
        layout.offsets.set_size(
            offsets_vec.size()
        );
        layout.event_counts.set_size(
            event_counts_vec.size()
        );
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            layout.order[i] =
                order_vec[i];
        }
        for (arma::uword i = 0;
                i < offsets_vec.size();
                ++i)
        {
            layout.offsets[i] =
                offsets_vec[i];
        }
        for (arma::uword i = 0;
                i < event_counts_vec.size();
                ++i)
        {
            layout.event_counts[i] =
                event_counts_vec[i];
        }
        layout.n_events = n_events;
        double first_event_time =
            std::numeric_limits<double>::infinity();
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            if (status_ptr[i] == 1.0)
            {
                first_event_time =
                    std::min(
                        first_event_time,
                        time_ptr[i]
                    );
            }
        }
        arma::uword n_active = 0;
        for (arma::uword i = 0;
                i < n;
                ++i)
        {
            if (time_ptr[i] >= first_event_time)
            {
                ++n_active;
            }
        }
        layout.n_active =
            n_active;
        return layout;
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
        const double *ptr =
            group_weight.memptr();
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            if (!std::isfinite(ptr[g]) ||
                    ptr[g] < 0.0)
            {
                Rcpp::stop(
                    "group_weight must be finite and "
                    "non-negative."
                );
            }
        }
    }


    inline double max_abs_matrix(
        const arma::mat& x
    )
    {
        const double *ptr =
            x.memptr();
        double result = 0.0;
        for (arma::uword i = 0;
                i < x.n_elem;
                ++i)
        {
            result = std::max(
                         result,
                         std::abs(ptr[i])
                     );
        }
        return result;
    }


    inline Rcpp::NumericMatrix copy_arma_matrix(
        const arma::mat& x
    )
    {
        Rcpp::NumericMatrix result(
            static_cast<R_xlen_t>(x.n_rows),
            static_cast<R_xlen_t>(x.n_cols)
        );
        const double *ptr =
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
                ) = ptr[
                        i + j * x.n_rows
                    ];
            }
        }
        return result;
    }

} // anonymous namespace


// [[Rcpp::export]]
Rcpp::List sgl_cox_fit_cpp(
    const arma::mat& X,
    const arma::mat& time,
    const arma::mat& status,
    const arma::mat& group_index,
    const arma::mat& group_weight,
    const arma::mat& initial_beta,
    const double lambda,
    const double alpha,
    const double step_size,
    const int max_iter,
    const double tol
)
{
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    check_column_matrix(
        time,
        "time"
    );
    check_column_matrix(
        status,
        "status"
    );
    check_column_matrix(
        initial_beta,
        "initial_beta"
    );
    if (X.n_rows == 0 ||
            X.n_cols == 0)
    {
        Rcpp::stop(
            "X must contain at least one row and one column."
        );
    }
    if (time.n_rows != X.n_rows ||
            status.n_rows != X.n_rows)
    {
        Rcpp::stop(
            "time and status must have nrow(X) rows."
        );
    }
    if (initial_beta.n_rows != X.n_cols)
    {
        Rcpp::stop(
            "nrow(initial_beta) must equal ncol(X)."
        );
    }
    check_binary_status(status);
    check_nonnegative_finite(
        lambda,
        "lambda"
    );
    check_unit_interval(
        alpha,
        "alpha"
    );
    check_positive_scalar(
        step_size,
        "step_size"
    );
    if (max_iter <= 0)
    {
        Rcpp::stop(
            "max_iter must be positive."
        );
    }
    check_positive_scalar(
        tol,
        "tol"
    );
    const sgl::GroupLayout group_layout =
        build_group_layout(group_index);
    validate_group_weight(
        group_weight,
        group_layout.offsets.n_elem - 1
    );
    const sgl::CoxRiskLayout risk_layout =
        build_cox_risk_layout(
            time,
            status
        );
    arma::mat beta =
        initial_beta;
    arma::mat eta(
        X.n_rows,
        1,
        arma::fill::none
    );
    sgl::initialize_cox_eta_inplace(
        eta,
        X,
        beta
    );
    arma::mat gradient(
        X.n_cols,
        1,
        arma::fill::none
    );
    sgl::CoxGradientWorkspace gradient_workspace(
        X.n_rows,
        X.n_cols
    );
    arma::mat beta_extrapolated =
        beta;
    arma::mat eta_extrapolated =
        eta;
    double fista_t =
        1.0;
    const double lambda_l1 =
        step_size * lambda * alpha;
    const double lambda_group =
        step_size *
        lambda *
        (1.0 - alpha);
    bool converged = false;
    int iterations = 0;
    for (int iter = 0;
            iter < max_iter;
            ++iter)
    {
        /*
         * The Cox gradient is computed on the extrapolated eta.
         */
        sgl::cox_breslow_gradient_objective_inplace(
            gradient,
            X,
            eta_extrapolated,
            status,
            risk_layout,
            gradient_workspace,
            false
        );
        /*
         * Proximal update of the extrapolated state.
         */
        sgl::sparse_group_proximal_gradient_update_eta_inplace(
            beta_extrapolated,
            eta_extrapolated,
            X,
            gradient,
            group_layout,
            group_weight,
            step_size,
            lambda_l1,
            lambda_group
        );
        const double fista_t_next =
            sgl::fista_next_t(
                fista_t
            );
        const double coefficient =
            (fista_t - 1.0) /
            fista_t_next;
        /*
         * Commit the new beta/eta,
         * and compute the extrapolated state for the next round.
         */
        const double beta_change =
            sgl::fista_commit_and_extrapolate_inplace(
                beta,
                beta_extrapolated,
                eta,
                eta_extrapolated,
                coefficient
            );
        const double scale =
            1.0 +
            max_abs_matrix(beta);
        fista_t =
            fista_t_next;
        iterations =
            iter + 1;
        if (beta_change <= tol * scale)
        {
            converged =
                true;
            break;
        }
    }
    const double objective =
        sgl::cox_sparse_group_objective(
            beta,
            X,
            eta,
            status,
            risk_layout,
            group_weight,
            group_layout,
            lambda,
            alpha
        );
    arma::mat risk_score(
        eta.n_rows,
        1,
        arma::fill::none
    );
    double max_eta =
        -std::numeric_limits<double>::infinity();
    for (arma::uword i = 0;
            i < eta.n_rows;
            ++i)
    {
        max_eta =
            std::max(
                max_eta,
                eta(i, 0)
            );
    }
    for (arma::uword i = 0;
            i < eta.n_rows;
            ++i)
    {
        risk_score(i, 0) =
            std::exp(
                eta(i, 0) - max_eta
            );
    }
    arma::mat intercept(
        1,
        1,
        arma::fill::zeros
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
               Rcpp::_["risk_score"] =
                   copy_arma_matrix(risk_score),
               Rcpp::_["objective"] =
                   objective,
               Rcpp::_["iterations"] =
                   iterations,
               Rcpp::_["converged"] =
                   converged
           );
}

// [[Rcpp::export]]
Rcpp::List sgl_cox_path_cpp(
    const arma::mat& X,
    const arma::mat& time,
    const arma::mat& status,
    const arma::mat& group_index,
    const arma::mat& group_weight,
    const arma::mat& initial_beta,
    const Rcpp::NumericVector& lambda_path,
    const double alpha,
    const double step_size,
    const int max_iter,
    const double tol
)
{
    if (!X.is_finite())
    {
        Rcpp::stop(
            "X must contain only finite values."
        );
    }
    check_column_matrix(
        time,
        "time"
    );
    check_column_matrix(
        status,
        "status"
    );
    check_column_matrix(
        initial_beta,
        "initial_beta"
    );
    if (X.n_rows == 0 ||
            X.n_cols == 0)
    {
        Rcpp::stop(
            "X must contain at least one row and "
            "one column."
        );
    }
    if (time.n_rows != X.n_rows ||
            status.n_rows != X.n_rows)
    {
        Rcpp::stop(
            "time and status must have nrow(X) rows."
        );
    }
    if (initial_beta.n_rows != X.n_cols)
    {
        Rcpp::stop(
            "nrow(initial_beta) must equal ncol(X)."
        );
    }
    check_binary_status(status);
    check_unit_interval(
        alpha,
        "alpha"
    );
    check_positive_scalar(
        step_size,
        "step_size"
    );
    if (max_iter <= 0)
    {
        Rcpp::stop(
            "max_iter must be positive."
        );
    }
    check_positive_scalar(
        tol,
        "tol"
    );
    const R_xlen_t n_lambda_r =
        lambda_path.size();
    if (n_lambda_r == 0)
    {
        Rcpp::stop(
            "lambda_path must contain at least "
            "one value."
        );
    }
    const arma::uword n_lambda =
        static_cast<arma::uword>(
            n_lambda_r
        );
    for (arma::uword lambda_index = 0;
            lambda_index < n_lambda;
            ++lambda_index)
    {
        check_nonnegative_finite(
            lambda_path[lambda_index],
            "lambda_path"
        );
    }
    /*
     * Key optimization 1:
     * The GroupLayout is built only once for the entire lambda path.
     */
    const sgl::GroupLayout group_layout =
        build_group_layout(
            group_index
        );
    validate_group_weight(
        group_weight,
        group_layout.offsets.n_elem - 1
    );
    /*
     * Key optimization 2:
     * time/status are unchanged, so the CoxRiskLayout
     * is built only once.
     */
    const sgl::CoxRiskLayout risk_layout =
        build_cox_risk_layout(
            time,
            status
        );
    /*
     * Compute lambdas from largest to smallest,
     * keeping the warm start.
     */
    std::vector<arma::uword> calculation_order(
        n_lambda
    );
    for (arma::uword lambda_index = 0;
            lambda_index < n_lambda;
            ++lambda_index)
    {
        calculation_order[lambda_index] =
            lambda_index;
    }
    std::stable_sort(
        calculation_order.begin(),
        calculation_order.end(),
        [&lambda_path](
            const arma::uword a,
            const arma::uword b
        )
    {
        return lambda_path[a] >
               lambda_path[b];
    }
    );
    arma::mat beta =
        initial_beta;
    arma::mat eta(
        X.n_rows,
        1,
        arma::fill::none
    );
    sgl::initialize_cox_eta_inplace(
        eta,
        X,
        beta
    );
    arma::mat gradient(
        X.n_cols,
        1,
        arma::fill::none
    );
    sgl::CoxGradientWorkspace gradient_workspace(
        X.n_rows,
        X.n_cols
    );
    arma::mat beta_path(
        X.n_cols,
        n_lambda,
        arma::fill::zeros
    );
    Rcpp::IntegerVector iterations(
        n_lambda
    );
    Rcpp::LogicalVector converged(
        n_lambda
    );
    arma::mat beta_extrapolated =
        beta;
    arma::mat eta_extrapolated =
        eta;
    double fista_t =
        1.0;
    for (arma::uword position = 0;
            position < n_lambda;
            ++position)
    {
        const arma::uword lambda_index =
            calculation_order[position];
        const double lambda =
            lambda_path[lambda_index];
        const double lambda_l1 =
            step_size *
            lambda *
            alpha;
        const double lambda_group =
            step_size *
            lambda *
            (1.0 - alpha);
        /*
         * The new lambda uses the previous lambda's beta
         * as a warm start, but does not inherit the previous lambda's
         * FISTA momentum.
         */
        beta_extrapolated =
            beta;
        eta_extrapolated =
            eta;
        fista_t =
            1.0;
        bool lambda_converged =
            false;
        int lambda_iterations =
            0;
        for (int iter = 0;
                iter < max_iter;
                ++iter)
        {
            /*
             * The gradient is computed on the extrapolated state.
             */
            sgl::cox_breslow_gradient_objective_inplace(
                gradient,
                X,
                eta_extrapolated,
                status,
                risk_layout,
                gradient_workspace,
                false
            );
            /*
             * The proximal update also acts on the
             * extrapolated state.
             */
            sgl::sparse_group_proximal_gradient_update_eta_inplace(
                beta_extrapolated,
                eta_extrapolated,
                X,
                gradient,
                group_layout,
                group_weight,
                step_size,
                lambda_l1,
                lambda_group
            );
            const double fista_t_next =
                sgl::fista_next_t(
                    fista_t
                );
            const double coefficient =
                (fista_t - 1.0) /
                fista_t_next;
            /*
             * Commit the proximal result to beta/eta,
             * and generate the extrapolated state for the next round.
             */
            const double beta_change =
                sgl::fista_commit_and_extrapolate_inplace(
                    beta,
                    beta_extrapolated,
                    eta,
                    eta_extrapolated,
                    coefficient
                );
            fista_t =
                fista_t_next;
            const double scale =
                1.0 +
                max_abs_matrix(beta);
            lambda_iterations =
                iter + 1;
            if (beta_change <= tol * scale)
            {
                lambda_converged =
                    true;
                break;
            }
        }
        beta_path.col(lambda_index) =
                     beta;
        iterations[lambda_index] =
            lambda_iterations;
        converged[lambda_index] =
            lambda_converged;
    }
    return Rcpp::List::create(
               Rcpp::_["beta_path"] =
                   copy_arma_matrix(beta_path),
               Rcpp::_["iterations"] =
                   iterations,
               Rcpp::_["converged"] =
                   converged
           );
}