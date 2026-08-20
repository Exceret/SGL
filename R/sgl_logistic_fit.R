#' Fit a Logistic sparse-group lasso model.
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


#' @export
print.sgl_logistic_fit <- function(x, ...) {
  message("Logistic sparse-group lasso fit\n")
  message("Objective:", format(x$objective), "\n")
  message("Iterations:", x$iterations, "\n")
  message("Converged:", x$converged, "\n")
  invisible(x)
}
