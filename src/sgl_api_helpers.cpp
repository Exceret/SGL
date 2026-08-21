#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

namespace
{

    inline void check_finite_vector(
    const Rcpp::NumericVector& x,
    const char *name
    )
    {
        for (R_xlen_t i = 0;
                i < x.size();
                ++i)
        {
            if (!std::isfinite(x[i]))
            {
                Rcpp::stop(
                    "%s must contain only finite values.",
                    name
                );
            }
        }
    }


    inline void check_finite_matrix(
        const arma::mat& x,
              const char *name
    )
    {
        if (!x.is_finite())
        {
            Rcpp::stop(
                "%s must contain only finite values.",
                name
            );
        }
    }


    inline arma::uvec validate_group_index(
        const arma::mat& group_index,
        const arma::uword p
    )
    {
        if (group_index.n_cols != 1 ||
                group_index.n_rows != p)
        {
            Rcpp::stop(
                "group_index must be a p x 1 matrix."
            );
        }
        check_finite_matrix(
            group_index,
            "group_index"
        );
        const double *group_ptr =
            group_index.memptr();
        arma::uvec groups(
            p,
            arma::fill::none
        );
        arma::uword n_groups = 0;
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            const double value =
                group_ptr[j];
            if (value < 1.0 ||
                    std::floor(value) != value)
            {
                Rcpp::stop(
                    "group_index must contain positive "
                    "integer labels."
                );
            }
            groups[j] =
                static_cast<arma::uword>(
                    value
                );
            n_groups =
                std::max(
                    n_groups,
                    groups[j]
                );
        }
        if (n_groups == 0)
        {
            Rcpp::stop(
                "group_index must contain at least "
                "one group."
            );
        }
        arma::uvec counts(
            n_groups,
            arma::fill::zeros
        );
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            ++counts[groups[j] - 1];
        }
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            if (counts[g] == 0)
            {
                Rcpp::stop(
                    "group_index must use consecutive "
                    "labels 1,...,G."
                );
            }
        }
        return groups;
    }


    inline void validate_group_weight(
        const arma::mat& group_weight,
        const arma::uword n_groups
    )
    {
        if (group_weight.n_cols != 1 ||
                group_weight.n_rows != n_groups)
        {
            Rcpp::stop(
                "group_weight must be a G x 1 matrix."
            );
        }
        check_finite_matrix(
            group_weight,
            "group_weight"
        );
        const double *weight_ptr =
            group_weight.memptr();
        for (arma::uword g = 0;
                g < n_groups;
                ++g)
        {
            if (weight_ptr[g] < 0.0)
            {
                Rcpp::stop(
                    "group_weight must be non-negative."
                );
            }
        }
    }

    inline double sparse_group_lambda_max_for_group(
        std::vector<double> correlations,
        const double alpha,
        const double group_weight
    )
    {
        if (correlations.empty())
        {
            return 0.0;
        }
        std::sort(
            correlations.begin(),
            correlations.end(),
            std::greater<double>()
        );
        const std::size_t group_size =
            correlations.size();
        /*
         * alpha = 1: lasso
         */
        if (alpha >= 1.0)
        {
            return correlations.front();
        }
        /*
         * alpha = 0: group lasso
         */
        if (alpha <= 0.0)
        {
            if (group_weight <= 0.0)
            {
                return 0.0;
            }
            double squared_norm = 0.0;
            for (const double value : correlations)
            {
                squared_norm += value * value;
            }
            return std::sqrt(squared_norm) /
                   group_weight;
        }
        /*
         * 如果 group penalty 权重为 0，
         * 该组退化为 lasso。
         */
        if (group_weight <= 0.0)
        {
            return correlations.front() / alpha;
        }
        const double group_penalty =
            (1.0 - alpha) * group_weight;
        double lambda_group_max = 0.0;
        double sum_correlations = 0.0;
        double sum_squared_correlations = 0.0;
        /*
         * 假设当前有 k 个变量经过 lasso soft-threshold
         * 后仍然处于 active 状态。
         *
         * 方程为：
         *
         *   sum_i (c_i - lambda * alpha)^2
         *       =
         *   lambda^2 * ((1-alpha) * w)^2
         */
        for (std::size_t k = 1;
                k <= group_size;
                ++k)
        {
            const double correlation =
                correlations[k - 1];
            sum_correlations +=
                correlation;
            sum_squared_correlations +=
                correlation * correlation;
            const double k_double =
                static_cast<double>(k);
            const double a =
                k_double * alpha * alpha -
                group_penalty * group_penalty;
            const double b =
                -2.0 * alpha *
                sum_correlations;
            const double c =
                sum_squared_correlations;
            /*
             * active set 为前 k 个变量时，
             * lambda 必须满足：
             *
             *   c[k] / alpha <= lambda <= c[k-1] / alpha
             */
            const double lower =
                (k < group_size)
                ? correlations[k] / alpha
                : 0.0;
            const double upper =
                correlations[k - 1] / alpha;
            const double interval_tolerance =
                1e-10 *
                std::max(
                    1.0,
                    upper
                );
            auto accept_root =
                [&](const double root)
            {
                if (!std::isfinite(root) ||
                        root < 0.0)
                {
                    return;
                }
                if (root >= lower - interval_tolerance &&
                        root <= upper + interval_tolerance)
                {
                    lambda_group_max =
                        std::max(
                            lambda_group_max,
                            root
                        );
                }
            };
            /*
             * A 接近 0 时退化为一元一次方程。
             */
            const double coefficient_tolerance =
                1e-14 *
                std::max(
                    1.0,
                    std::max(
                        std::abs(a),
                        std::max(
                            std::abs(b),
                            std::abs(c)
                        )
                    )
                );
            if (std::abs(a) <= coefficient_tolerance)
            {
                if (std::abs(b) > coefficient_tolerance)
                {
                    accept_root(-c / b);
                }
                continue;
            }
            double discriminant =
                b * b -
                4.0 * a * c;
            /*
             * 允许极小的浮点误差。
             */
            const double discriminant_tolerance =
                1e-12 *
                std::max(
                    1.0,
                    std::max(
                        b * b,
                        std::abs(
                            4.0 * a * c
                        )
                    )
                );
            if (discriminant < 0.0 &&
                    discriminant >=
                    -discriminant_tolerance)
            {
                discriminant = 0.0;
            }
            if (discriminant < 0.0)
            {
                continue;
            }
            const double sqrt_discriminant =
                std::sqrt(discriminant);
            accept_root(
                (-b + sqrt_discriminant) /
                (2.0 * a)
            );
            accept_root(
                (-b - sqrt_discriminant) /
                (2.0 * a)
            );
        }
        return lambda_group_max;
    }

} // anonymous namespace


// [[Rcpp::export]]
double sgl_lambda_max_from_gradient_cpp(
    const arma::mat& gradient,
    const arma::mat& group_index,
    const arma::mat& group_weight,
    const double alpha
)
{
    if (gradient.n_cols != 1)
    {
        Rcpp::stop(
            "gradient must be a single-column matrix."
        );
    }
    if (!std::isfinite(alpha) ||
            alpha < 0.0 ||
            alpha > 1.0)
    {
        Rcpp::stop(
            "alpha must be in [0, 1]."
        );
    }
    const arma::uword p =
        gradient.n_rows;
    check_finite_matrix(
        gradient,
        "gradient"
    );
    const arma::uvec groups =
        validate_group_index(
            group_index,
            p
        );
    const arma::uword n_groups =
        group_weight.n_rows;
    validate_group_weight(
        group_weight,
        n_groups
    );
    if (n_groups != groups.max())
    {
        Rcpp::stop(
            "group_weight has an incompatible "
            "number of rows."
        );
    }
    const double *gradient_ptr = gradient.memptr();
    const double *weight_ptr = group_weight.memptr();
    double lambda_max = 0.0;
    for (arma::uword g = 1;
            g <= n_groups;
            ++g)
    {
        std::vector<double> correlations;
        for (arma::uword j = 0;
                j < p;
                ++j)
        {
            if (groups[j] == g)
            {
                correlations.push_back(
                    std::abs(
                        gradient_ptr[j]
                    )
                );
            }
        }
        const double group_lambda =
            sparse_group_lambda_max_for_group(
                correlations,
                alpha,
                weight_ptr[g - 1]
            );
        lambda_max =
            std::max(
                lambda_max,
                group_lambda
            );
    }
    if (!std::isfinite(lambda_max) ||
            lambda_max <= 0.0)
    {
        lambda_max =
            std::numeric_limits<double>::epsilon();
    }
    return lambda_max;
}


// [[Rcpp::export]]
Rcpp::NumericVector sgl_make_lambda_path_cpp(
    const Rcpp::Nullable<Rcpp::NumericVector> lambda_input,
    const double lambda_max,
    const int n_lambda,
    const double min_frac,
    const double lambda_decay
)
{
    if (n_lambda <= 0)
    {
        Rcpp::stop(
            "n_lambda must be positive."
        );
    }
    if (!std::isfinite(lambda_max) ||
            lambda_max < 0.0)
    {
        Rcpp::stop(
            "lambda_max must be finite and "
            "non-negative."
        );
    }
    if (!std::isfinite(min_frac) ||
            min_frac <= 0.0 ||
            min_frac > 1.0)
    {
        Rcpp::stop(
            "min_frac must be in (0, 1]."
        );
    }
    /*
     * 该参数只为兼容现有 Rcpp 接口保留。
     * main/master 的 lambda path 不使用 gamma。
     */
    if (!std::isfinite(lambda_decay) ||
            lambda_decay <= 0.0 ||
            lambda_decay >= 1.0)
    {
        Rcpp::stop(
            "lambda_decay must be in (0, 1)."
        );
    }
    if (lambda_input.isNotNull())
    {
        SEXP lambda_sexp =
            lambda_input.get();
        Rcpp::NumericVector supplied(
            lambda_sexp
        );
        if (supplied.size() == 0)
        {
            Rcpp::stop(
                "lambda_input must not be empty."
            );
        }
        check_finite_vector(
            supplied,
            "lambda_input"
        );
        for (R_xlen_t i = 0;
                i < supplied.size();
                ++i)
        {
            if (supplied[i] < 0.0)
            {
                Rcpp::stop(
                    "lambda_input must be "
                    "non-negative."
                );
            }
        }
        return Rcpp::clone(
                   supplied
               );
    }
    const double safe_lambda_max =
        std::max(
            lambda_max,
            std::numeric_limits<double>::epsilon()
        );
    if (n_lambda == 1)
    {
        return Rcpp::NumericVector::create(
                   safe_lambda_max
               );
    }
    const double lambda_min =
        std::max(
            safe_lambda_max * min_frac,
            std::numeric_limits<double>::epsilon()
        );
    Rcpp::NumericVector result(
        n_lambda
    );
    const double log_max =
        std::log(safe_lambda_max);
    const double log_min =
        std::log(lambda_min);
    /*
     * 与 main/master 一致：
     *
     *   lambda_max,
     *   ...,
     *   lambda_max * min_frac
     *
     * gamma 不参与 lambda path。
     */
    for (int i = 0;
            i < n_lambda;
            ++i)
    {
        const double fraction =
            static_cast<double>(i) /
            static_cast<double>(n_lambda - 1);
        result[i] =
            std::exp(
                log_max +
                fraction *
                (log_min - log_max)
            );
    }
    return result;
}


// [[Rcpp::export]]
Rcpp::List sgl_parse_cox_response_cpp(
    const arma::mat& response
)
{
    if (response.n_cols != 2)
    {
        Rcpp::stop(
            "Cox response must be an n x 2 matrix: "
            "time and status."
        );
    }
    if (response.n_rows == 0)
    {
        Rcpp::stop(
            "Cox response must contain at least "
            "one observation."
        );
    }
    if (!response.is_finite())
    {
        Rcpp::stop(
            "Cox response must contain only finite values."
        );
    }
    const arma::uword n =
        response.n_rows;
    arma::mat time(
        n,
        1,
        arma::fill::none
    );
    arma::mat status(
        n,
        1,
        arma::fill::none
    );
    bool has_event = false;
    for (arma::uword i = 0;
            i < n;
            ++i)
    {
        const double time_i =
            response(i, 0);
        const double status_i =
            response(i, 1);
        if (status_i != 0.0 &&
                status_i != 1.0)
        {
            Rcpp::stop(
                "Cox status must contain only 0 and 1."
            );
        }
        time(i, 0) =
            time_i;
        status(i, 0) =
            status_i;
        if (status_i == 1.0)
        {
            has_event = true;
        }
    }
    if (!has_event)
    {
        Rcpp::stop(
            "Cox response must contain at least one event."
        );
    }
    return Rcpp::List::create(
               Rcpp::_["time"] = time,
               Rcpp::_["status"] = status
           );
}


// [[Rcpp::export]]
Rcpp::NumericMatrix sgl_cox_zero_gradient_cpp(
    const arma::mat& X,
    const arma::mat& time,
    const arma::mat& status
)
{
    if (time.n_cols != 1 ||
            status.n_cols != 1)
    {
        Rcpp::stop(
            "time and status must be single-column matrices."
        );
    }
    if (time.n_rows != X.n_rows ||
            status.n_rows != X.n_rows)
    {
        Rcpp::stop(
            "time and status must have nrow(X) rows."
        );
    }
    check_finite_matrix(
        X,
        "X"
    );
    check_finite_matrix(
        time,
        "time"
    );
    check_finite_matrix(
        status,
        "status"
    );
    const arma::uword n =
        X.n_rows;
    const arma::uword p =
        X.n_cols;
    const double *time_ptr =
        time.memptr();
    const double *status_ptr =
        status.memptr();
    std::vector<arma::uword> order(
        n
    );
    for (arma::uword i = 0;
            i < n;
            ++i)
    {
        if (status_ptr[i] != 0.0 &&
                status_ptr[i] != 1.0)
        {
            Rcpp::stop(
                "status must contain only 0 and 1."
            );
        }
        order[i] = i;
    }
    std::stable_sort(
        order.begin(),
        order.end(),
        [&time_ptr](
            const arma::uword a,
            const arma::uword b
        )
    {
        return time_ptr[a] >
               time_ptr[b];
    }
    );
    arma::mat gradient(
        p,
        1,
        arma::fill::zeros
    );
    // const double *gradient_x_ptr =
    //     X.memptr();
    arma::uword n_events = 0;
    double risk_count = 0.0;
    arma::rowvec risk_sum(
        p,
        arma::fill::zeros
    );
    arma::uword begin = 0;
    while (begin < n)
    {
        const double current_time =
            time_ptr[order[begin]];
        arma::uword end =
            begin + 1;
        while (
            end < n &&
            time_ptr[order[end]] ==
            current_time
        )
        {
            ++end;
        }
        /*
         * 当前时间组加入风险集。
         * 因为 order 按 time 降序，
         * 累积风险集满足 time >= current_time。
         */
        for (arma::uword k = begin;
                k < end;
                ++k)
        {
            const arma::uword i =
                order[k];
            risk_count += 1.0;
            for (arma::uword j = 0;
                    j < p;
                    ++j)
            {
                risk_sum[j] +=
                    X(i, j);
            }
        }
        arma::uword event_count = 0;
        arma::rowvec event_sum(
            p,
            arma::fill::zeros
        );
        for (arma::uword k = begin;
                k < end;
                ++k)
        {
            const arma::uword i =
                order[k];
            if (status_ptr[i] == 1.0)
            {
                ++event_count;
                ++n_events;
                for (arma::uword j = 0;
                        j < p;
                        ++j)
                {
                    event_sum[j] +=
                        X(i, j);
                }
            }
        }
        if (event_count > 0)
        {
            for (arma::uword j = 0;
                    j < p;
                    ++j)
            {
                gradient(j, 0) +=
                    static_cast<double>(
                        event_count
                    ) *
                    risk_sum[j] /
                    risk_count -
                    event_sum[j];
            }
        }
        begin = end;
    }
    if (n_events == 0)
    {
        Rcpp::stop(
            "status must contain at least one event."
        );
    }
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
    if (n_active == 0)
    {
        Rcpp::stop(
            "Cox response must contain at least "
            "one active observation."
        );
    }
    gradient /=
        static_cast<double>(
            n_active
        );    return Rcpp::wrap(
                         gradient
                     );
}

