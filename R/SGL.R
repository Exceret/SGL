#' Fit a sparse-group lasso model
#'
#' Fits a regularized generalized linear model via penalized maximum
#' likelihood with a combination of lasso and group lasso penalties.  The
#' model is fit for a path of values of the penalty parameter \code{lambda}.
#' Fits linear, logistic and Cox models.  The SGL penalty is
#' \deqn{\lambda (\alpha \|\beta\|_1 + (1 - \alpha) \sum_g w_g \|\beta_g\|_2)}{lambda * (alpha * ||beta||_1 + (1 - alpha) * sum_g w_g ||beta_g||_2)}
#' where the weights \eqn{w_g} are the square roots of the group sizes.
#'
#' @param data For \code{type = "linear"} should be a list with \code{x} an
#'   input matrix of dimension n-obs by p-vars, and \code{y} a length n
#'   response vector.  For \code{type = "logit"} should be a list with
#'   \code{x}, an input matrix as before, and \code{y} a length n binary
#'   response vector.  For \code{type = "cox"} should be a list with
#'   \code{x} as before, \code{time}, an n-vector of failure/censor times,
#'   and \code{status}, an n-vector indicating failure (1) or censoring (0).
#' @param index A p-vector indicating group membership of each covariate.
#' @param type Model type: one of \code{"linear"}, \code{"logit"},
#'   \code{"cox"}.
#' @param maxit Maximum number of iterations to convergence.
#' @param thresh Convergence threshold for change in beta.
#' @param min.frac The minimum value of the penalty parameter, as a fraction
#'   of the maximum value.
#' @param nlam Number of lambda values to use in the regularization path.
#' @param gamma Fitting parameter used for tuning backtracking (between 0 and
#'   1).
#' @param standardize Logical flag for variable standardization prior to
#'   fitting the model.
#' @param verbose Logical flag for whether or not step number will be output.
#' @param step Fitting parameter used for initial backtracking step size
#'   (between 0 and 1).
#' @param reset Fitting parameter used for taking advantage of local strong
#'   convexity in Nesterov momentum (number of iterations before momentum
#'   term is reset).
#' @param alpha The mixing parameter.  \code{alpha = 1} is the lasso penalty;
#'   \code{alpha = 0} is the group lasso penalty.
#' @param lambdas A user specified sequence of lambda values for fitting.  We
#'   recommend leaving this \code{NULL} and letting \code{SGL} self-select
#'   values.
#' @param .preprocessed internal flag for whether or not the data has been
#'   preprocessed.
#'
#' @return An object with S3 class \code{"SGL"}:
#'   \item{beta}{A p by nlam matrix of coefficient estimates.}
#'   \item{lambdas}{The actual list of lambda values used in the
#'     regularization path.}
#'   \item{type}{Response type (\code{"linear"}, \code{"logit"} or
#'     \code{"cox"}).}
#'   \item{intercept}{The intercept(s) of the fitted model.}
#'   \item{X.transform}{A list with components \code{X.means} and
#'     \code{X.scale} used to standardize new data for prediction.}
#'
#' @references Simon, N., Friedman, J., Hastie, T., and Tibshirani, R. (2011)
#'   \emph{A Sparse-Group Lasso}, Journal of Computational and Graphical
#'   Statistics, 22(2), 231--245.
#'
#' @examples
#' set.seed(1)
#' n <- 50; p <- 100; size.groups <- 10
#' index <- ceiling(1:p / size.groups)
#' X <- matrix(rnorm(n * p), ncol = p, nrow = n)
#' beta <- (-2:2)
#' y <- X[, 1:5] %*% beta + 0.1 * rnorm(n)
#' data <- list(x = X, y = y)
#' fit <- SGL(data, index, type = "linear")
#' print(fit)
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
  lambdas = NULL,
  .preprocessed = FALSE
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

  if (!all(is.finite(data$x))) {
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
    !all(is.finite(index_values)) ||
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
        !all(is.finite(lambdas)) ||
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

  if (
    length(.preprocessed) != 1L ||
      !is.logical(.preprocessed) ||
      is.na(.preprocessed)
  ) {
    stop(".preprocessed must be one logical value.")
  }

  if (isTRUE(.preprocessed)) {
    X_fit <- X

    X.transform <- list(
      X.means = rep(0, p),
      X.scale = rep(1, p)
    )
  } else {
    transformed <- sgl_center_scale_cpp(
      X,
      standardize
    )

    X_fit <- transformed$x
    X_transform <- transformed$X.transform

    X.transform <- list(
      X.means = X_transform[, 1L],
      X.scale = if (isTRUE(standardize)) {
        X_transform[, 2L]
      } else {
        1
      }
    )
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
        !all(is.finite(y))
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

    lambda_max <- SGL_lambda_max_from_gradient(
      gradient = initial_gradient,
      group_index = group_index,
      group_weight = group_weight,
      alpha = alpha
    )

    lambda_path <- SGL_make_lambda_path(
      lambdas = lambda_path,
      lambda_max = lambda_max,
      nlam = nlam,
      min_frac = min.frac,
      gamma = gamma
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

      beta_path[, lambda_index] <- fit$beta[, 1L]

      beta_start <- matrix(
        0,
        nrow = p,
        ncol = 1L
      )

      intercept_start <- initial_intercept

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
      X.transform = X.transform
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
        !all(is.finite(y)) ||
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

    initial_intercept <- stats::qlogis(initial_probability)

    m_y <- mean(y)

    resp <- m_y * m_y * (1 - m_y) - (y - m_y)

    initial_gradient <- crossprod(X_fit, resp) / n

    lambda_max <- SGL_lambda_max_from_gradient(
      gradient = initial_gradient,
      group_index = group_index,
      group_weight = group_weight,
      alpha = alpha
    )

    lambda_path <- SGL_make_lambda_path(
      lambdas = lambda_path,
      lambda_max = lambda_max,
      nlam = nlam,
      min_frac = min.frac,
      gamma = gamma
    )

    step_size <- step / (1 + 0.25 * sum(X_fit * X_fit) / n)

    n_lambda <- length(lambda_path)

    calculation_order <- order(
      lambda_path,
      decreasing = TRUE,
      method = "radix"
    )

    path_fit <- sgl_logistic_path_cpp(
      X = X_fit,
      y = y_matrix,
      group_index = group_index,
      group_weight = group_weight,
      initial_beta = matrix(
        0,
        nrow = p,
        ncol = 1L
      ),
      initial_intercept = matrix(
        initial_intercept,
        nrow = 1L,
        ncol = 1L
      ),
      lambda_path = lambda_path,
      alpha = as.numeric(alpha),
      step_size = as.numeric(step_size),
      max_iter = as.integer(maxit),
      tol = as.numeric(thresh),
      fit_intercept = TRUE
    )

    beta_path <- path_fit$beta_path

    intercept_path <- path_fit$intercept_path

    if (isTRUE(verbose)) {
      for (position in seq_along(calculation_order)) {
        lambda_index <-
          calculation_order[position]

        message(
          "logit lambda[",
          lambda_index,
          "] = ",
          format(lambda_path[lambda_index]),
          ", iterations = ",
          path_fit$iterations[lambda_index],
          ", converged = ",
          path_fit$converged[lambda_index]
        )
      }
    }
    result <- list(
      beta = beta_path,
      lambdas = lambda_path,
      type = "logit",
      intercept = intercept_path,
      X.transform = X.transform
    )
  } else {
    if (is.null(data$y)) {
      data$y <- cbind(data$time, data$status)
    }

    if (!is.matrix(data$y)) {
      stop(
        "data$y must be a matrix for type = 'cox'."
      )
    }

    cox <- sgl_parse_cox_response_cpp(data$y)

    time <- cox$time[, 1L]
    status <- cox$status[, 1L]

    initial_beta <- numeric(p)

    initial_gradient <- SGL_cox_zero_gradient(
      X = X_fit,
      time = time,
      status = status
    )

    lambda_max <- SGL_lambda_max_from_gradient(
      gradient = initial_gradient,
      group_index = group_index,
      group_weight = group_weight,
      alpha = alpha
    )

    lambda_path <- SGL_make_lambda_path(
      lambdas = lambda_path,
      lambda_max = lambda_max,
      nlam = nlam,
      min_frac = min.frac,
      gamma = gamma
    )

    first_event_time <- min(time[status == 1])

    n_active <- sum(time >= first_event_time)

    step_size <- step / (1 + sum(X_fit * X_fit) / n_active)

    calculation_order <- order(
      lambda_path,
      decreasing = TRUE,
      method = "radix"
    )

    n_lambda <- length(lambda_path)

    path_fit <- sgl_cox_path_cpp(
      X = X_fit,
      time = matrix(
        time,
        ncol = 1L
      ),
      status = matrix(
        status,
        ncol = 1L
      ),
      group_index = group_index,
      group_weight = group_weight,
      initial_beta = matrix(
        0,
        nrow = p,
        ncol = 1L
      ),
      lambda_path = as.numeric(lambda_path),
      alpha = as.numeric(alpha),
      step_size = as.numeric(step_size),
      max_iter = as.integer(maxit),
      tol = as.numeric(thresh)
    )

    beta_path <-
      path_fit$beta_path

    if (isTRUE(verbose)) {
      for (position in seq_along(calculation_order)) {
        lambda_index <-
          calculation_order[position]

        message(
          "cox lambda[",
          lambda_index,
          "] = ",
          format(lambda_path[lambda_index]),
          ", iterations = ",
          path_fit$iterations[lambda_index],
          ", converged = ",
          path_fit$converged[lambda_index]
        )
      }
    }

    result <- list(
      beta = beta_path,
      lambdas = lambda_path,
      type = "cox",
      X.transform = X.transform
    )
  }

  class(result) <- "SGL"

  result
}
