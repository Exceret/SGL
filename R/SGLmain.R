#' Fit a sparse-group lasso model.
#'
#' @param data A list containing the data.
#' @param index A numeric vector containing the group labels.
#' @param type The type of model to fit.
#' @param maxit The maximum number of iterations.
#' @param thresh The convergence threshold.
#' @param min.frac The minimum fraction of non-zero coefficients.
#' @param nlam The number of lambda values to try.
#' @param gamma The gamma value.
#' @param standardize Whether to standardize the data.
#' @param verbose Whether to print progress messages.
#' @param step The step size.
#' @param reset The number of iterations between resets.
#' @param alpha The alpha value.
#' @param lambdas The lambda values to try.
#'
#' @export
SGL <- function(
  data,
  index,
  type = "linear",
  maxit = 1000,
  thresh = 0.001,
  min.frac = 0.1,
  nlam = 20,
  gamma = 0.8,
  standardize = TRUE,
  verbose = FALSE,
  step = 1,
  reset = 10,
  alpha = 0.95,
  lambdas = NULL
) {
  if (!(type %in% c("linear", "logit", "cox"))) {
    print(
      "'type' must be one of 'linear', 'logit', or 'cox'!"
    )
    return(NA)
  }

  if (!is.list(data)) {
    stop("data must be a list.")
  }

  if (
    is.null(data$x) ||
      !is.matrix(data$x) ||
      !is.numeric(data$x)
  ) {
    stop("data$x must be a numeric matrix.")
  }

  if (any(!is.finite(data$x))) {
    stop("data$x must contain only finite values.")
  }

  X <- data$x
  n <- nrow(X)
  p <- ncol(X)

  if (n == 0L || p == 0L) {
    stop(
      "data$x must contain at least one row and one column."
    )
  }

  if (length(index) != p) {
    stop(
      "length(index) must equal ncol(data$x)."
    )
  }

  index_values <- as.numeric(index)

  if (
    any(!is.finite(index_values)) ||
      any(index_values < 1) ||
      any(index_values != floor(index_values))
  ) {
    stop(
      "index must contain positive integer group labels."
    )
  }

  unique_groups <- sort(unique(index_values))

  group_id <- match(
    index_values,
    unique_groups
  )

  group_index <- matrix(
    as.integer(group_id),
    ncol = 1L
  )

  n_groups <- length(unique_groups)

  group_weight <- matrix(
    sqrt(tabulate(
      group_id,
      nbins = n_groups
    )),
    ncol = 1L
  )

  if (
    length(maxit) != 1L ||
      is.na(maxit) ||
      maxit <= 0 ||
      maxit != as.integer(maxit)
  ) {
    stop("maxit must be one positive integer.")
  }

  if (
    length(thresh) != 1L ||
      !is.finite(thresh) ||
      thresh <= 0
  ) {
    stop("thresh must be one finite positive value.")
  }

  if (
    length(min.frac) != 1L ||
      !is.finite(min.frac) ||
      min.frac <= 0 ||
      min.frac > 1
  ) {
    stop("min.frac must be in (0, 1].")
  }

  if (
    length(nlam) != 1L ||
      is.na(nlam) ||
      nlam <= 0 ||
      nlam != as.integer(nlam)
  ) {
    stop("nlam must be one positive integer.")
  }

  if (
    length(gamma) != 1L ||
      !is.finite(gamma) ||
      gamma <= 0 ||
      gamma >= 1
  ) {
    stop("gamma must be in (0, 1).")
  }

  if (
    length(alpha) != 1L ||
      !is.finite(alpha) ||
      alpha < 0 ||
      alpha > 1
  ) {
    stop("alpha must be in [0, 1].")
  }

  if (
    length(step) != 1L ||
      !is.finite(step) ||
      step <= 0
  ) {
    stop("step must be finite and positive.")
  }

  if (
    length(reset) != 1L ||
      is.na(reset) ||
      reset < 0 ||
      reset != as.integer(reset)
  ) {
    stop("reset must be a non-negative integer.")
  }

  if (!is.null(lambdas)) {
    if (
      !is.numeric(lambdas) ||
        length(lambdas) == 0L ||
        any(!is.finite(lambdas)) ||
        any(lambdas < 0)
    ) {
      stop(
        "lambdas must contain finite non-negative values."
      )
    }

    lambda_path <- as.numeric(lambdas)
  } else {
    lambda_path <- NULL
  }

  invisible(reset)

  transformed <- center_scale(
    X,
    standardize
  )

  X_fit <- transformed$x
  X_transform <- transformed$X.transform

  lambda_max_from_gradient <- function(
    gradient
  ) {
    gradient <- as.numeric(gradient)

    value <- 0

    if (alpha > 0) {
      value <- max(
        value,
        max(abs(gradient)) / alpha
      )
    }

    if (alpha < 1) {
      for (g in seq_len(n_groups)) {
        selected <- which(
          group_index[, 1L] == g
        )

        weight <- group_weight[g, 1L]

        if (weight > 0) {
          group_value <- sqrt(
            sum(gradient[selected]^2)
          ) /
            ((1 - alpha) * weight)

          value <- max(
            value,
            group_value
          )
        }
      }
    }

    max(
      value,
      .Machine$double.eps
    )
  }

  make_lambda_path <- function(
    supplied_lambda,
    lambda_max
  ) {
    if (!is.null(supplied_lambda)) {
      return(as.numeric(supplied_lambda))
    }

    lambda_max <- max(
      as.numeric(lambda_max),
      .Machine$double.eps
    )

    if (nlam == 1L) {
      return(lambda_max)
    }

    lambda_min <- max(
      lambda_max * min.frac,
      .Machine$double.eps
    )

    candidate <- lambda_max *
      gamma^(0:(nlam - 1L))

    if (tail(candidate, 1L) > lambda_min) {
      candidate <- exp(seq(
        log(lambda_max),
        log(lambda_min),
        length.out = nlam
      ))
    }

    as.numeric(candidate)
  }

  parse_cox_response <- function(
    response
  ) {
    if (inherits(response, "Surv")) {
      response_matrix <- as.matrix(response)

      if (ncol(response_matrix) != 2L) {
        stop(
          "Only right-censored Cox responses are supported."
        )
      }

      time <- response_matrix[, 1L]
      status <- response_matrix[, 2L]
    } else if (
      is.matrix(response) &&
        ncol(response) == 2L
    ) {
      time <- response[, 1L]
      status <- response[, 2L]
    } else if (
      is.list(response) &&
        !is.null(response$time) &&
        !is.null(response$status)
    ) {
      time <- response$time
      status <- response$status
    } else {
      stop(
        "For type = 'cox', data$y must be a Surv object, a two-column matrix, or a list with time and status."
      )
    }

    time <- as.numeric(time)
    status <- as.numeric(status)

    if (
      length(time) != n ||
        length(status) != n
    ) {
      stop(
        "Cox time and status must have nrow(data$x) values."
      )
    }

    if (
      any(!is.finite(time)) ||
        any(!is.finite(status))
    ) {
      stop(
        "Cox time and status must contain only finite values."
      )
    }

    if (!all(status %in% c(0, 1))) {
      stop(
        "Cox status must contain only 0 and 1."
      )
    }

    if (!any(status == 1)) {
      stop(
        "Cox response must contain at least one event."
      )
    }

    list(
      time = matrix(time, ncol = 1L),
      status = matrix(status, ncol = 1L)
    )
  }

  cox_zero_gradient <- function(
    X,
    time,
    status
  ) {
    event_times <- sort(
      unique(time[status == 1]),
      decreasing = TRUE
    )

    gradient <- numeric(
      ncol(X)
    )

    for (event_time in event_times) {
      event_index <- which(
        time == event_time &
          status == 1
      )

      risk_index <- which(
        time >= event_time
      )

      risk_mean <- colMeans(
        X[risk_index, , drop = FALSE]
      )

      event_sum <- colSums(
        X[event_index, , drop = FALSE]
      )

      gradient <- gradient +
        length(event_index) * risk_mean -
        event_sum
    }

    gradient / sum(status == 1)
  }

  if (identical(type, "linear")) {
    if (
      is.null(data$y) ||
        !is.numeric(data$y)
    ) {
      stop(
        "data$y must be numeric for type = 'linear'."
      )
    }

    y <- as.numeric(data$y)

    if (
      length(y) != n ||
        any(!is.finite(y))
    ) {
      stop(
        "data$y must contain n finite values."
      )
    }

    y_matrix <- matrix(
      y,
      ncol = 1L
    )

    initial_intercept <- matrix(
      mean(y),
      nrow = 1L,
      ncol = 1L
    )

    initial_residual <- matrix(
      initial_intercept[1L, 1L] - y,
      ncol = 1L
    )

    initial_gradient <- crossprod(
      X_fit,
      initial_residual
    ) /
      n

    lambda_max <- lambda_max_from_gradient(
      initial_gradient
    )

    lambda_path <- make_lambda_path(
      lambda_path,
      lambda_max
    )

    step_size <- step / (1 + sum(X_fit * X_fit) / n)

    calculation_order <- order(
      lambda_path,
      decreasing = TRUE,
      method = "radix"
    )

    n_lambda <- length(lambda_path)

    beta_path <- matrix(
      0,
      nrow = p,
      ncol = n_lambda
    )

    fits <- vector(
      mode = "list",
      length = n_lambda
    )

    beta_start <- matrix(
      0,
      nrow = p,
      ncol = 1L
    )

    intercept_start <- initial_intercept

    for (position in seq_along(calculation_order)) {
      lambda_index <- calculation_order[position]

      fit <- sgl_linear_fit(
        X = X_fit,
        y = y_matrix,
        group_index = group_index,
        group_weight = group_weight,
        lambda = lambda_path[lambda_index],
        alpha = alpha,
        step_size = step_size,
        max_iter = as.integer(maxit),
        tol = thresh,
        fit_intercept = TRUE,
        initial_beta = beta_start,
        initial_intercept = intercept_start
      )

      fits[[lambda_index]] <- fit

      beta_path[, lambda_index] <-
        fit$beta[, 1L]

      beta_start <- fit$beta
      intercept_start <- fit$intercept

      if (isTRUE(verbose)) {
        message(
          "linear lambda[",
          lambda_index,
          "] = ",
          format(lambda_path[lambda_index]),
          ", iterations = ",
          fit$iterations,
          ", converged = ",
          fit$converged
        )
      }
    }

    result <- list(
      beta = beta_path,
      lambdas = lambda_path,
      type = "linear",
      intercept = mean(y),
      X.transform = X_transform
    )
  } else if (identical(type, "logit")) {
    if (
      is.null(data$y) ||
        !is.numeric(data$y)
    ) {
      stop(
        "data$y must be numeric for type = 'logit'."
      )
    }

    y <- as.numeric(data$y)

    if (
      length(y) != n ||
        any(!is.finite(y)) ||
        !all(y %in% c(0, 1))
    ) {
      stop(
        "data$y must contain only 0 and 1 for type = 'logit'."
      )
    }

    y_matrix <- matrix(
      y,
      ncol = 1L
    )

    initial_probability <- min(
      max(mean(y), 1e-8),
      1 - 1e-8
    )

    initial_intercept <- qlogis(
      initial_probability
    )

    initial_gradient <- crossprod(
      X_fit,
      initial_probability - y
    ) /
      n

    lambda_max <- lambda_max_from_gradient(
      initial_gradient
    )

    lambda_path <- make_lambda_path(
      lambda_path,
      lambda_max
    )

    step_size <- step / (1 + 0.25 * sum(X_fit * X_fit) / n)

    calculation_order <- order(
      lambda_path,
      decreasing = TRUE,
      method = "radix"
    )

    n_lambda <- length(lambda_path)

    beta_path <- matrix(
      0,
      nrow = p,
      ncol = n_lambda
    )

    intercept_path <- numeric(
      n_lambda
    )

    fits <- vector(
      mode = "list",
      length = n_lambda
    )

    beta_start <- matrix(
      0,
      nrow = p,
      ncol = 1L
    )

    intercept_start <- matrix(
      initial_intercept,
      nrow = 1L,
      ncol = 1L
    )

    for (position in seq_along(calculation_order)) {
      lambda_index <- calculation_order[position]

      fit <- sgl_logistic_fit(
        X = X_fit,
        y = y_matrix,
        group_index = group_index,
        group_weight = group_weight,
        lambda = lambda_path[lambda_index],
        alpha = alpha,
        step_size = step_size,
        max_iter = as.integer(maxit),
        tol = thresh,
        fit_intercept = TRUE,
        initial_beta = beta_start,
        initial_intercept = intercept_start
      )

      fits[[lambda_index]] <- fit

      beta_path[, lambda_index] <-
        fit$beta[, 1L]

      intercept_path[lambda_index] <-
        fit$intercept[1L, 1L]

      beta_start <- fit$beta
      intercept_start <- fit$intercept

      if (isTRUE(verbose)) {
        message(
          "logit lambda[",
          lambda_index,
          "] = ",
          format(lambda_path[lambda_index]),
          ", iterations = ",
          fit$iterations,
          ", converged = ",
          fit$converged
        )
      }
    }

    result <- list(
      beta = beta_path,
      lambdas = lambda_path,
      type = "logit",
      intercept = intercept_path,
      X.transform = X_transform
    )
  } else {
    cox <- parse_cox_response(
      data$y
    )

    time <- cox$time[, 1L]
    status <- cox$status[, 1L]

    initial_beta <- numeric(
      p
    )

    initial_gradient <- cox_zero_gradient(
      X = X_fit,
      time = time,
      status = status
    )

    lambda_max <- lambda_max_from_gradient(
      initial_gradient
    )

    lambda_path <- make_lambda_path(
      lambda_path,
      lambda_max
    )

    n_events <- sum(
      status == 1
    )

    step_size <- step / (1 + sum(X_fit * X_fit) / n_events)

    calculation_order <- order(
      lambda_path,
      decreasing = TRUE,
      method = "radix"
    )

    n_lambda <- length(lambda_path)

    beta_path <- matrix(
      0,
      nrow = p,
      ncol = n_lambda
    )

    fits <- vector(
      mode = "list",
      length = n_lambda
    )

    beta_start <- matrix(
      0,
      nrow = p,
      ncol = 1L
    )

    for (position in seq_along(calculation_order)) {
      lambda_index <- calculation_order[position]

      fit <- sgl_cox_fit(
        X = X_fit,
        time = matrix(time, ncol = 1L),
        status = matrix(status, ncol = 1L),
        group_index = group_index,
        group_weight = group_weight,
        lambda = lambda_path[lambda_index],
        alpha = alpha,
        step_size = step_size,
        max_iter = as.integer(maxit),
        tol = thresh,
        initial_beta = beta_start
      )

      fits[[lambda_index]] <- fit

      beta_path[, lambda_index] <-
        fit$beta[, 1L]

      beta_start <- fit$beta

      if (isTRUE(verbose)) {
        message(
          "cox lambda[",
          lambda_index,
          "] = ",
          format(lambda_path[lambda_index]),
          ", iterations = ",
          fit$iterations,
          ", converged = ",
          fit$converged
        )
      }
    }

    result <- list(
      beta = beta_path,
      lambdas = lambda_path,
      type = "cox",
      X.transform = X_transform
    )
  }

  class(result) <- "SGL"

  result
}

center_scale <- function(X, standardize) {
  means <- apply(X, 2, mean)
  X <- t(t(X) - means)

  X.transform <- list(X.means = means)

  if (standardize == TRUE) {
    var <- apply(X, 2, function(x) (sqrt(sum(x^2))))
    X <- t(t(X) / var)
    X.transform$X.scale <- var
  } else {
    X.transform$X.scale <- 1
  }

  return(list(x = X, X.transform = X.transform))
}
# form_folds <- function(n_obs, nfold) {
#   folds <- cut(seq(1, n_obs), breaks = nfold, labels = FALSE)
#   folds_final <- sample(folds, replace = FALSE)
#   return(folds_final)
# }
