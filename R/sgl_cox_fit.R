#' Fit a Cox proportional hazards model with a sparse-group lasso penalty
#'
#' Fits a Cox proportional hazards regression model with the sparse-group
#' lasso penalty for a single value of the penalty parameter \code{lambda}.
#' Ties in event times are handled with the Breslow approximation.  No
#' ordinary intercept is fitted.
#'
#' The partial log-likelihood is maximized subject to the penalty
#' \deqn{\lambda\left(\alpha \|\beta\|_1 +
#' (1 - \alpha)\sum_g w_g \|\beta_g\|_2\right),}{%
#' lambda * (alpha * ||b||_1 + (1 - alpha) * sum_g w_g ||b_g||_2),}
#' with group weights \eqn{w_g = \sqrt{|g|}} by default.
#'
#' @usage
#' sgl_cox_fit(X, time, status, group_index, lambda, alpha = 1,
#'   group_weight = NULL, step_size = NULL, max_iter = 1000L, tol = 1e-8,
#'   initial_beta = NULL)
#'
#' @param X A numeric design matrix with \code{n} rows and \code{p} columns.
#' @param time A numeric single-column matrix with \code{n} rows containing
#'   the failure/censoring times.
#' @param status A numeric single-column matrix with \code{n} rows containing
#'   the event indicators: 1 for an observed event and 0 for censoring.  At
#'   least one event is required.
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
#' @param initial_beta Optional p x 1 matrix of starting coefficient values.
#'
#' @return A list with class \code{"sgl_cox_fit"} containing the fitted
#'   \code{beta}, the linear \code{prediction} and \code{eta}, the final
#'   \code{objective} (the negative Breslow partial log-likelihood), the
#'   number of \code{iterations} and a logical \code{converged} flag.
#'
#' @seealso \code{\link{SGL}}, \code{\link{sgl_linear_fit}},
#'   \code{\link{sgl_logistic_fit}}
#'
#' @examples
#' set.seed(1)
#' n <- 60; p <- 12
#' X <- matrix(rnorm(n * p), nrow = n, ncol = p)
#' time <- matrix(rexp(n), ncol = 1)
#' status <- matrix(rbinom(n, 1, 0.6), ncol = 1)
#' group_index <- matrix(rep(1:4, each = 3), ncol = 1)
#' fit <- sgl_cox_fit(X, time, status, group_index, lambda = 0.05)
#' print(fit)
#'
#' @export
sgl_cox_fit <- function(
  X,
  time,
  status,
  group_index,
  lambda,
  alpha = 1,
  group_weight = NULL,
  step_size = NULL,
  max_iter = 1000L,
  tol = 1e-8,
  initial_beta = NULL
) {
  if (!is.matrix(X) || !is.numeric(X)) {
    stop("X must be a numeric matrix.")
  }

  if (!all(is.finite(X))) {
    stop("X must contain only finite values.")
  }

  if (
    !is.matrix(time) ||
      ncol(time) != 1L ||
      !is.numeric(time)
  ) {
    stop("time must be a numeric single-column matrix.")
  }

  if (
    !is.matrix(status) ||
      ncol(status) != 1L ||
      !is.numeric(status)
  ) {
    stop(
      "status must be a numeric single-column matrix."
    )
  }

  if (!all(is.finite(time))) {
    stop("time must contain only finite values.")
  }

  if (
    !all(is.finite(status)) ||
      !all(status[, 1L] %in% c(0, 1))
  ) {
    stop("status must contain only 0 and 1.")
  }

  n <- nrow(X)
  p <- ncol(X)

  if (n == 0L || p == 0L) {
    stop(
      "X must contain at least one row and one column."
    )
  }

  if (
    nrow(time) != n ||
      nrow(status) != n
  ) {
    stop(
      "time and status must have nrow(X) rows."
    )
  }

  if (sum(status[, 1L] == 1) == 0L) {
    stop("status must contain at least one event.")
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
    !is.matrix(group_index) ||
      ncol(group_index) != 1L ||
      length(groups) != p
  ) {
    stop(
      "group_index must be a p x 1 matrix."
    )
  }

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

    if (
      !all(is.finite(group_weight)) ||
        any(group_weight < 0)
    ) {
      stop(
        "group_weight must be finite and non-negative."
      )
    }
  }

  if (is.null(initial_beta)) {
    initial_beta <- matrix(
      0,
      nrow = p,
      ncol = 1L
    )
  } else {
    if (
      !is.matrix(initial_beta) ||
        ncol(initial_beta) != 1L ||
        nrow(initial_beta) != p
    ) {
      stop(
        "initial_beta must be a p x 1 matrix."
      )
    }

    if (!all(is.finite(initial_beta))) {
      stop(
        "initial_beta must contain only finite values."
      )
    }
  }

  if (is.null(step_size)) {
    n_events <- sum(status[, 1L] == 1)

    step_size <- 1 / (1 + sum(X * X) / n_events)
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

  result <- sgl_cox_fit_cpp(
    X = X,
    time = time,
    status = status,
    group_index = group_index,
    group_weight = group_weight,
    initial_beta = initial_beta,
    lambda = as.numeric(lambda),
    alpha = as.numeric(alpha),
    step_size = as.numeric(step_size),
    max_iter = as.integer(max_iter),
    tol = as.numeric(tol)
  )

  class(result) <- "sgl_cox_fit"

  result
}


#' @param x An object of class \code{"sgl_cox_fit"}, as returned by
#'   \code{\link{sgl_cox_fit}}.
#' @param ... Additional arguments passed to other methods (currently
#'   unused).
#'
#' @export
#' @method print sgl_cox_fit
#' @rdname sgl_cox_fit
print.sgl_cox_fit <- function(x, ...) {
  message("Cox sparse-group lasso fit\n")
  message("Ties method: Breslow\n")
  message("Objective:", format(x$objective), "\n")
  message("Iterations:", x$iterations, "\n")
  message("Converged:", x$converged, "\n")
  invisible(x)
}
