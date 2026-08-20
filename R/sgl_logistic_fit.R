#' Fit a logistic sparse-group lasso model
#'
#' Fits a logistic regression model with the sparse-group lasso penalty for a
#' single value of the penalty parameter \code{lambda}.  The optimized
#' objective is
#' \deqn{-\frac{1}{n}\sum_{i=1}^n
#' \left[y_i \log p_i + (1 - y_i) \log(1 - p_i)\right] +
#' \lambda\left(\alpha \|\beta\|_1 +
#' (1 - \alpha)\sum_g w_g \|\beta_g\|_2\right),}{%
#' -1/n * sum_i [y_i log p_i + (1-y_i) log(1-p_i)] +
#' lambda * (alpha * ||b||_1 + (1 - alpha) * sum_g w_g ||b_g||_2),}
#' where \eqn{p_i} is the fitted probability for observation \eqn{i} and the
#' group weights \eqn{w_g = \sqrt{|g|}} by default.
#'
#' @usage
#' sgl_logistic_fit(X, y, group_index, lambda, alpha = 1,
#'   group_weight = NULL, step_size = NULL, max_iter = 1000L, tol = 1e-8,
#'   fit_intercept = TRUE, initial_beta = NULL, initial_intercept = NULL)
#'
#' @param X A numeric design matrix with \code{n} rows and \code{p} columns.
#' @param y A single-column numeric matrix with \code{n} rows containing the
#'   binary response, with values in \code{{0, 1}}.
#' @param group_index A single-column matrix of length \code{p} giving the
#'   group label (a positive integer between 1 and G) of each predictor.
#' @param lambda One finite non-negative penalty value.
#' @param alpha The mixing parameter between 0 and 1.  \code{alpha = 1} is
#'   the lasso penalty, \code{alpha = 0} is the group lasso penalty.
#' @param group_weight Optional G x 1 matrix of group weights.  Defaults to
#'   the square roots of the group sizes.
#' @param step_size Optional positive step size for the proximal gradient
#'   updates.  Defaults to the reciprocal of an upper bound on the Lipschitz
#'   constant of the gradient.
#' @param max_iter Maximum number of iterations.
#' @param tol Convergence tolerance on the change in the objective.
#' @param fit_intercept Logical; whether to fit an intercept.
#' @param initial_beta Optional p x 1 matrix of starting coefficient values.
#' @param initial_intercept Optional 1 x 1 matrix of starting intercept
#'   values.
#'
#' @return A list with class \code{"sgl_logistic_fit"} containing the fitted
#'   \code{beta} and \code{intercept}, the linear \code{prediction} and
#'   \code{eta}, the final \code{objective}, the number of \code{iterations}
#'   and a logical \code{converged} flag.
#'
#' @seealso \code{\link{SGL}}, \code{\link{sgl_linear_fit}},
#'   \code{\link{sgl_cox_fit}}
#'
#' @examples
#' set.seed(1)
#' n <- 60; p <- 12
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' y <- matrix(rbinom(n, 1, 0.4), ncol = 1)
#' group_index <- matrix(rep(1:4, each = 3), ncol = 1)
#' fit <- sgl_logistic_fit(X, y, group_index, lambda = 0.05)
#' print(fit)
#'
#' @export
sgl_logistic_fit <- function(
  X,
  y,
  group_index,
  lambda,
  alpha = 1,
  group_weight = NULL,
  step_size = NULL,
  max_iter = 1000L,
  tol = 1e-8,
  fit_intercept = TRUE,
  initial_beta = NULL,
  initial_intercept = NULL
) {
  if (
    !is.matrix(X) ||
      !is.numeric(X)
  ) {
    stop("X must be a numeric matrix.")
  }

  if (
    !is.matrix(y) ||
      ncol(y) != 1L
  ) {
    stop("y must be a single-column matrix.")
  }

  if (!all(y[, 1L] %in% c(0, 1))) {
    stop("y must contain only 0 and 1.")
  }

  if (
    !is.matrix(group_index) ||
      ncol(group_index) != 1L
  ) {
    stop(
      "group_index must be a single-column matrix."
    )
  }

  n <- nrow(X)
  p <- ncol(X)

  if (n == 0L) {
    stop("X must contain at least one observation.")
  }

  if (p == 0L) {
    stop("X must contain at least one variable.")
  }

  if (nrow(y) != n) {
    stop("nrow(y) must equal nrow(X).")
  }

  if (nrow(group_index) != p) {
    stop(
      "nrow(group_index) must equal ncol(X)."
    )
  }

  if (
    length(lambda) != 1L ||
      !is.finite(lambda) ||
      lambda < 0
  ) {
    stop(
      "lambda must be one finite non-negative value."
    )
  }

  if (
    length(alpha) != 1L ||
      !is.finite(alpha) ||
      alpha < 0 ||
      alpha > 1
  ) {
    stop("alpha must be in [0, 1].")
  }

  groups <- as.numeric(
    group_index[, 1L]
  )

  if (
    !all(is.finite(groups)) ||
      any(groups < 1) ||
      any(groups != floor(groups))
  ) {
    stop(
      "group_index must contain positive integers."
    )
  }

  labels <- sort(unique(groups))
  n_groups <- max(labels)

  if (
    length(labels) != n_groups ||
      any(labels != seq_len(n_groups))
  ) {
    stop(
      "group_index must use consecutive labels 1,...,G."
    )
  }

  if (is.null(group_weight)) {
    group_weight <- matrix(
      sqrt(tabulate(
        groups,
        nbins = n_groups
      )),
      ncol = 1L
    )
  } else {
    if (
      !is.matrix(group_weight) ||
        ncol(group_weight) != 1L ||
        nrow(group_weight) != n_groups
    ) {
      stop(
        "group_weight must be a G x 1 matrix."
      )
    }
  }

  if (is.null(initial_beta)) {
    initial_beta <- matrix(
      0,
      nrow = p,
      ncol = 1L
    )
  } else if (
    !is.matrix(initial_beta) ||
      ncol(initial_beta) != 1L ||
      nrow(initial_beta) != p
  ) {
    stop(
      "initial_beta must be a p x 1 matrix."
    )
  }

  if (is.null(initial_intercept)) {
    initial_intercept <- matrix(
      0,
      nrow = 1L,
      ncol = 1L
    )
  } else if (
    !is.matrix(initial_intercept) ||
      !identical(dim(initial_intercept), c(1L, 1L))
  ) {
    stop(
      "initial_intercept must be a 1 x 1 matrix."
    )
  }

  if (is.null(step_size)) {
    lipschitz_upper_bound <- 1 +
      0.25 * sum(X * X) / n

    step_size <- 1 /
      lipschitz_upper_bound
  }

  if (
    length(step_size) != 1L ||
      !is.finite(step_size) ||
      step_size <= 0
  ) {
    stop(
      "step_size must be one finite positive value."
    )
  }

  result <- sgl_logistic_fit_cpp(
    X = X,
    y = y,
    group_index = group_index,
    group_weight = group_weight,
    initial_beta = initial_beta,
    initial_intercept = initial_intercept,
    lambda = as.numeric(lambda),
    alpha = as.numeric(alpha),
    step_size = as.numeric(step_size),
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol),
    fit_intercept = isTRUE(fit_intercept)
  )

  class(result) <- "sgl_logistic_fit"

  result
}


#' @param x An object of class \code{"sgl_logistic_fit"}, as returned by
#'   \code{\link{sgl_logistic_fit}}.
#' @param ... Additional arguments passed to other methods (currently
#'   unused).
#'
#' @export
#' @method print sgl_logistic_fit
#' @rdname sgl_logistic_fit
print.sgl_logistic_fit <- function(x, ...) {
  message("Logistic sparse-group lasso fit\n")
  message("Objective:", format(x$objective), "\n")
  message("Iterations:", x$iterations, "\n")
  message("Converged:", x$converged, "\n")
  invisible(x)
}
