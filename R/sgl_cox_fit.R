#' Fit a Cox sparse-group lasso model.
#'
#' Cox uses the Breslow approximation for tied event times.
#' No ordinary intercept is fitted.
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

  if (any(!is.finite(X))) {
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

  if (any(!is.finite(time))) {
    stop("time must contain only finite values.")
  }

  if (
    any(!is.finite(status)) ||
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
    any(!is.finite(groups)) ||
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
      any(!is.finite(group_weight)) ||
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

    if (any(!is.finite(initial_beta))) {
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


#' @export
print.sgl_cox_fit <- function(x, ...) {
  cat("Cox sparse-group lasso fit\n")
  cat("Ties method: Breslow\n")
  cat("Objective:", format(x$objective), "\n")
  cat("Iterations:", x$iterations, "\n")
  cat("Converged:", x$converged, "\n")
  invisible(x)
}
