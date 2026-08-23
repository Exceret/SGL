#' Fit and cross-validate a sparse-group lasso model
#'
#' @param data A list containing x and the response.
#' @param index A vector indicating group membership of each variable.
#' @param type One of "linear", "logit", or "cox".
#' @param maxit Maximum number of iterations.
#' @param thresh Convergence threshold.
#' @param min.frac Minimum lambda as a fraction of lambda max.
#' @param nlam Number of lambda values.
#' @param gamma Lambda-path decay parameter.
#' @param nfold Number of cross-validation folds.
#' @param standardize Whether to center and scale predictors.
#' @param verbose Whether to print progress.
#' @param step Initial proximal-gradient step-size parameter.
#' @param reset Kept for API compatibility.
#' @param alpha Sparse/group lasso mixing parameter.
#' @param lambdas Optional user-supplied lambda sequence.
#' @param foldid Optional fold assignment vector.
#'
#' @return An object of class \code{"cvSGL"}.
#'
#' @export
cvSGL <- function(
  data,
  index = rep(1L, ncol(data$x)),
  type = "linear",
  maxit = 1000L,
  thresh = 0.001,
  min.frac = 0.05,
  nlam = 20L,
  gamma = 0.8,
  nfold = 10L,
  standardize = TRUE,
  verbose = FALSE,
  step = 1L,
  reset = 10L,
  alpha = 0.95,
  lambdas = NULL,
  foldid = NULL
) {
  if (
    !is.character(type) ||
      length(type) != 1L ||
      !type %in% c("linear", "logit", "cox")
  ) {
    stop(
      "'type' must be one of 'linear', 'logit', or 'cox'."
    )
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

  if (
    length(index) != p ||
      !is.numeric(index) ||
      !all(is.finite(index)) ||
      any(index < 1L) ||
      any(index != floor(index))
  ) {
    stop(
      "index must contain one positive integer label per variable."
    )
  }

  if (
    length(maxit) != 1L ||
      is.na(maxit) ||
      maxit <= 0L ||
      maxit != as.integer(maxit)
  ) {
    stop("maxit must be one positive integer.")
  }

  if (
    length(thresh) != 1L ||
      !is.finite(thresh) ||
      thresh <= 0L
  ) {
    stop("thresh must be one finite positive value.")
  }

  if (
    length(min.frac) != 1L ||
      !is.finite(min.frac) ||
      min.frac <= 0L ||
      min.frac > 1L
  ) {
    stop("min.frac must be in (0, 1].")
  }

  if (
    length(nlam) != 1L ||
      is.na(nlam) ||
      nlam <= 0L ||
      nlam != as.integer(nlam)
  ) {
    stop("nlam must be one positive integer.")
  }

  if (
    length(gamma) != 1L ||
      !is.finite(gamma) ||
      gamma <= 0L ||
      gamma >= 1L
  ) {
    stop("gamma must be in (0, 1).")
  }

  if (
    length(nfold) != 1L ||
      is.na(nfold) ||
      nfold < 2L ||
      nfold != as.integer(nfold)
  ) {
    stop("nfold must be an integer greater than or equal to 2.")
  }

  nfold <- as.integer(nfold)

  if (nfold > n) {
    stop("nfold cannot be greater than the number of observations.")
  }

  if (length(standardize) != 1L || is.na(standardize)) {
    stop("standardize must be one logical value.")
  }

  if (length(verbose) != 1L || is.na(verbose)) {
    stop("verbose must be one logical value.")
  }

  if (
    length(step) != 1L ||
      !is.finite(step) ||
      step <= 0L
  ) {
    stop("step must be finite and positive.")
  }

  if (
    length(reset) != 1L ||
      is.na(reset) ||
      reset < 0L ||
      reset != as.integer(reset)
  ) {
    stop("reset must be a non-negative integer.")
  }

  if (
    length(alpha) != 1L ||
      !is.finite(alpha) ||
      alpha < 0L ||
      alpha > 1L
  ) {
    stop("alpha must be in [0, 1].")
  }

  if (!is.null(lambdas)) {
    if (
      !is.numeric(lambdas) ||
        length(lambdas) == 0L ||
        !all(is.finite(lambdas)) ||
        any(lambdas < 0L)
    ) {
      stop(
        "lambdas must contain finite non-negative values."
      )
    }

    lambdas <- as.numeric(lambdas)
  }

  if (identical(type, "linear")) {
    if (is.null(data$y) || !is.numeric(data$y)) {
      stop(
        "data$y must be numeric for type = 'linear'."
      )
    }

    y <- as.numeric(data$y)

    if (length(y) != n || !all(is.finite(y))) {
      stop(
        "data$y must contain n finite values."
      )
    }

    data_cv <- list(
      x = X,
      y = y
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
        !all(y %in% c(0L, 1L))
    ) {
      stop(
        "data$y must contain n binary values."
      )
    }

    if (length(unique(y)) < 2L) {
      stop(
        "data$y must contain both classes."
      )
    }

    data_cv <- list(
      x = X,
      y = y
    )
  } else {
    if (!is.null(data$y)) {
      if (
        !is.matrix(data$y) ||
          ncol(data$y) != 2L ||
          nrow(data$y) != n
      ) {
        stop(
          "data$y must be an n x 2 matrix for type = 'cox'."
        )
      }

      time <- as.numeric(data$y[, 1L])
      status <- as.numeric(data$y[, 2L])
    } else {
      if (is.null(data$time) || is.null(data$status)) {
        stop(
          "Cox data must contain data$y or data$time "
        )
      }

      time <- as.numeric(data$time)
      status <- as.numeric(data$status)
    }

    if (length(time) != n || !all(is.finite(time))) {
      stop(
        "Cox time must contain n finite values."
      )
    }

    if (
      length(status) != n ||
        !all(is.finite(status)) ||
        !all(status %in% c(0L, 1L))
    ) {
      stop(
        "Cox status must contain n binary values."
      )
    }

    if (!any(status == 1L)) {
      stop(
        "Cox status must contain at least one event."
      )
    }

    data_cv <- list(
      x = X,
      time = time,
      status = status
    )
  }

  if (is.null(foldid)) {
    foldid <- cvSGL_make_foldid(
      n = n,
      nfold = nfold
    )
  } else {
    if (
      length(foldid) != n ||
        !is.numeric(foldid) ||
        !all(is.finite(foldid)) ||
        any(foldid != floor(foldid))
    ) {
      stop(
        "foldid must contain one integer fold label per observation."
      )
    }

    nfold <- length(unique(foldid))

    if (
      !identical(
        sort(unique(foldid)),
        seq_len(nfold)
      )
    ) {
      stop(
        "foldid labels must be 1, 2, ..., nfold."
      )
    }
  }

  if (identical(type, "logit")) {
    for (fold in seq_len(nfold)) {
      train <- foldid != fold

      if (length(unique(y[train])) < 2L) {
        stop(
          "Each logistic training fold must contain both classes."
        )
      }
    }
  }

  if (identical(type, "cox")) {
    for (fold in seq_len(nfold)) {
      train <- foldid != fold

      if (!any(status[train] == 1L)) {
        stop(
          "Each Cox training fold must contain at least one event."
        )
      }
    }
  }

  fit <- SGL(
    data = data_cv,
    index = index,
    type = type,
    maxit = as.integer(maxit),
    thresh = as.numeric(thresh),
    min.frac = as.numeric(min.frac),
    nlam = as.integer(nlam),
    gamma = as.numeric(gamma),
    standardize = isTRUE(standardize),
    verbose = isTRUE(verbose),
    step = as.numeric(step),
    reset = as.integer(reset),
    alpha = as.numeric(alpha),
    lambdas = lambdas,
    .preprocessed = FALSE
  )

  lambda_path <- as.numeric(
    fit$lambdas
  )

  n_lambda <- length(
    lambda_path
  )

  lldiffFold <- matrix(
    NA_real_,
    nrow = n_lambda,
    ncol = nfold
  )

  prevals <- matrix(
    NA_real_,
    nrow = n,
    ncol = n_lambda
  )

  # ------------------------------------------------------------
  # K-fold cross-validation
  # ------------------------------------------------------------

  for (fold in seq_len(nfold)) {
    ind.out <- foldid == fold
    ind.in <- !ind.out

    X_train_raw <- X[
      ind.in,
      ,
      drop = FALSE
    ]

    X_out_raw <- X[
      ind.out,
      ,
      drop = FALSE
    ]

    transformed_fold <- SGL_transform_train_test(
      X_train = X_train_raw,
      X_test = X_out_raw,
      standardize = isTRUE(standardize)
    )

    X_train <- transformed_fold$x_train
    X_out <- transformed_fold$x_test

    X_fold <- matrix(
      NA_real_,
      nrow = n,
      ncol = p
    )

    X_fold[ind.in, ] <- X_train
    X_fold[ind.out, ] <- X_out

    new_data <- cvSGL_subset_data(
      data_cv = data_cv,
      keep = ind.in,
      type = type
    )

    new_data$x <- X_train

    new_fit <- SGL(
      data = new_data,
      index = index,
      type = type,
      maxit = as.integer(maxit),
      thresh = as.numeric(thresh),
      min.frac = as.numeric(min.frac),
      nlam = n_lambda,
      gamma = as.numeric(gamma),
      standardize = isTRUE(standardize),
      verbose = FALSE,
      step = as.numeric(step),
      reset = as.integer(reset),
      alpha = as.numeric(alpha),
      lambdas = lambda_path,
      .preprocessed = TRUE
    )

    eta_all <- cvSGL_eta_cpp(fit = new_fit, X = X_fold, type = type)

    if (identical(type, "linear")) {
      y_out <- y[ind.out]

      for (lambda_index in seq_len(n_lambda)) {
        eta_out <- eta_all[ind.out, lambda_index]

        loss <- 0.5 * (y_out - eta_out)^2L

        lldiffFold[lambda_index, fold] <- sum(loss)

        prevals[ind.out, lambda_index] <- eta_out
      }
    } else if (identical(type, "logit")) {
      y_out <- y[ind.out]

      for (lambda_index in seq_len(n_lambda)) {
        eta_out <- eta_all[ind.out, lambda_index]

        loss <- pmax(eta_out, 0L) -
          y_out * eta_out +
          log1p(
            exp(-abs(eta_out))
          )

        lldiffFold[lambda_index, fold] <- sum(loss)

        prevals[ind.out, lambda_index] <- stats::plogis(eta_out)
      }
    } else {
      eta_train <- eta_all[ind.in, , drop = FALSE]

      for (lambda_index in seq_len(n_lambda)) {
        eta_current <- eta_all[, lambda_index]

        full_loss <- cvSGL_cox_negative_loglik(
          time = time,
          status = status,
          eta = eta_current
        )

        train_loss <- cvSGL_cox_negative_loglik(
          time = time[ind.in],
          status = status[ind.in],
          eta = eta_train[,
            lambda_index
          ]
        )

        lldiffFold[lambda_index, fold] <- full_loss - train_loss

        prevals[ind.out, lambda_index] <- eta_current[ind.out]
      }
    }

    if (isTRUE(verbose)) {
      message(
        "*** NFOLD ",
        fold,
        " / ",
        nfold,
        " ***"
      )
    }
  }
  # fit$X.transform <- full_X_transform
  lldiff <- if (requireNamespace("matrixStats", quietly = TRUE)) {
    matrixStats::rowSums2(lldiffFold)
  } else {
    rowSums(lldiffFold)
  }

  llSD <- if (requireNamespace("matrixStats", quietly = TRUE)) {
    matrixStats::rowSds(lldiffFold) * sqrt(nfold)
  } else {
    apply(lldiffFold, 1L, stats::sd) * sqrt(nfold)
  }

  lambda_min_index <- which.min(lldiff)

  lambda_min <- lambda_path[lambda_min_index]

  one_se_limit <- lldiff[lambda_min_index] + llSD[lambda_min_index]

  eligible <- which(lldiff <= one_se_limit)

  lambda_1se_index <- eligible[which.max(lambda_path[eligible])]

  lambda_1se <- lambda_path[lambda_1se_index]

  result <- list(
    lldiff = as.numeric(lldiff),
    llSD = as.numeric(llSD),
    lambdas = lambda_path,
    type = type,
    fit = fit,
    prevals = prevals,
    foldid = as.integer(foldid),
    lambda.min = lambda_min,
    lambda.min.index = lambda_min_index,
    lambda.1se = lambda_1se,
    lambda.1se.index = lambda_1se_index,
    lldiffFold = lldiffFold
  )

  class(result) <- "cvSGL"

  result
}

cvSGL_make_foldid <- function(
  n,
  nfold
) {
  folds <- cut(
    seq(1L, n),
    breaks = nfold,
    labels = FALSE
  )

  sample(
    folds,
    replace = FALSE
  )
}

cvSGL_subset_data <- function(
  data_cv,
  keep,
  type
) {
  if (identical(type, "cox")) {
    return(
      list(
        x = data_cv$x[keep, , drop = FALSE],
        time = data_cv$time[keep],
        status = data_cv$status[keep]
      )
    )
  }

  list(
    x = data_cv$x[keep, , drop = FALSE],
    y = data_cv$y[keep]
  )
}

cvSGL_cox_negative_loglik <- function(
  time,
  status,
  eta
) {
  event_times <- sort(unique(time[status == 1L]))

  if (!length(event_times)) {
    return(0L)
  }

  objective <- 0L

  for (event_time in event_times) {
    event <- (status == 1L & time == event_time)

    risk <- (time >= event_time)

    event_count <- sum(event)

    objective <- objective +
      event_count * cvSGL_cox_logsumexp(eta[risk]) - # cpp
      sum(eta[event])
  }

  first_event_time <- min(event_times)

  n_active <- sum(
    time >= first_event_time
  )

  if (n_active <= 0L) {
    return(0L)
  }

  objective / n_active
}
