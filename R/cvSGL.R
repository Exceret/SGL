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
#' @return An object of class \code{"cv.SGL"}.
#'
#' @export
cvSGL <- function(
  data,
  index = rep(1, ncol(data$x)),
  type = "linear",
  maxit = 1000,
  thresh = 0.001,
  min.frac = 0.05,
  nlam = 20,
  gamma = 0.8,
  nfold = 10,
  standardize = TRUE,
  verbose = FALSE,
  step = 1,
  reset = 10,
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
      any(index < 1) ||
      any(index != floor(index))
  ) {
    stop(
      "index must contain one positive integer label per variable."
    )
  }

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
    length(nfold) != 1L ||
      is.na(nfold) ||
      nfold < 2 ||
      nfold != as.integer(nfold)
  ) {
    stop("nfold must be an integer greater than or equal to 2.")
  }

  nfold <- as.integer(nfold)

  if (nfold > n) {
    stop("nfold cannot be greater than the number of observations.")
  }

  if (
    length(standardize) != 1L ||
      is.na(standardize)
  ) {
    stop("standardize must be one logical value.")
  }

  if (
    length(verbose) != 1L ||
      is.na(verbose)
  ) {
    stop("verbose must be one logical value.")
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

  if (
    length(alpha) != 1L ||
      !is.finite(alpha) ||
      alpha < 0 ||
      alpha > 1
  ) {
    stop("alpha must be in [0, 1].")
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

    lambdas <- as.numeric(lambdas)
  }

  # ------------------------------------------------------------
  # 统一整理响应数据
  # ------------------------------------------------------------

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
        !all(y %in% c(0, 1))
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
      if (
        is.null(data$time) ||
          is.null(data$status)
      ) {
        stop(
          "Cox data must contain data$y or data$time "
        )
      }

      time <- as.numeric(data$time)
      status <- as.numeric(data$status)
    }

    if (
      length(time) != n ||
        !all(is.finite(time))
    ) {
      stop(
        "Cox time must contain n finite values."
      )
    }

    if (
      length(status) != n ||
        !all(is.finite(status)) ||
        !all(status %in% c(0, 1))
    ) {
      stop(
        "Cox status must contain n binary values."
      )
    }

    if (!any(status == 1)) {
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

  # ------------------------------------------------------------
  # 生成或检查 fold
  # ------------------------------------------------------------

  .cvSGL_make_foldid <- function(
    type,
    n,
    nfold,
    y = NULL,
    status = NULL
  ) {
    if (identical(type, "logit")) {
      counts <- table(y)

      if (any(counts < nfold)) {
        stop(
          "Each logistic class must contain at least nfold "
        )
      }

      strata <- split(
        seq_len(n),
        y
      )
    } else if (identical(type, "cox")) {
      n_events <- sum(status == 1)

      if (n_events < nfold) {
        stop(
          "The number of Cox events must be at least nfold."
        )
      }

      strata <- split(
        seq_len(n),
        status
      )
    } else {
      strata <- list(seq_len(n))
    }

    foldid <- integer(n)

    for (current_stratum in strata) {
      current_stratum <-
        sample(current_stratum)

      fold_values <- rep(
        seq_len(nfold),
        length.out = length(current_stratum)
      )

      foldid[current_stratum] <-
        fold_values
    }

    foldid
  }

  if (is.null(foldid)) {
    foldid <- .cvSGL_make_foldid(
      type = type,
      n = n,
      nfold = nfold,
      y = if (identical(type, "logit")) y else NULL,
      status = if (identical(type, "cox")) status else NULL
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

    fold_labels <- sort(unique(foldid))

    if (length(fold_labels) < 2L) {
      stop("foldid must define at least two folds.")
    }

    foldid <- match(
      foldid,
      fold_labels
    )

    nfold <- length(fold_labels)
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

      if (!any(status[train] == 1)) {
        stop(
          "Each Cox training fold must contain at least one event."
        )
      }
    }
  }

  # ------------------------------------------------------------
  # 辅助函数
  # ------------------------------------------------------------

  .cvSGL_subset_data <- function(
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

  .cvSGL_eta <- function(
    fit,
    X
  ) {
    X <- as.matrix(X)

    X_means <- as.numeric(
      fit$X.transform$X.means
    )

    X_scale <- fit$X.transform$X.scale

    if (is.null(X_scale)) {
      X_scale <- rep(
        1,
        ncol(X)
      )
    }

    X_scale <- rep(
      as.numeric(X_scale),
      length.out = ncol(X)
    )

    X_new <- sweep(
      X,
      2L,
      X_means,
      "-"
    )

    X_new <- sweep(
      X_new,
      2L,
      X_scale,
      "/"
    )

    eta <- X_new %*% fit$beta

    if (!identical(type, "cox")) {
      eta <- sweep(
        eta,
        2L,
        as.numeric(fit$intercept),
        "+"
      )
    }

    eta
  }

  .cvSGL_cox_logsumexp <- function(x) {
    if (!length(x)) {
      return(-Inf)
    }

    maximum <- max(x)

    maximum +
      log(
        sum(
          exp(x - maximum)
        )
      )
  }

  .cvSGL_cox_negative_loglik <- function(
    time,
    status,
    eta
  ) {
    event_times <- sort(
      unique(
        time[status == 1]
      )
    )

    if (!length(event_times)) {
      return(0)
    }

    objective <- 0

    for (event_time in event_times) {
      event <- (status == 1 &
        time == event_time)

      risk <- (time >= event_time)

      event_count <- sum(event)

      objective <- objective +
        event_count *
          .cvSGL_cox_logsumexp(
            eta[risk]
          ) -
        sum(
          eta[event]
        )
    }

    objective
  }

  # ------------------------------------------------------------
  # 先对完整数据拟合一次，确定共同 lambda path
  # ------------------------------------------------------------

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
    lambdas = lambdas
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

    new_data <- .cvSGL_subset_data(
      data_cv = data_cv,
      keep = ind.in,
      type = type
    )

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
      lambdas = lambda_path
    )

    eta_all <- .cvSGL_eta(
      fit = new_fit,
      X = X
    )

    if (identical(type, "linear")) {
      y_out <- y[ind.out]

      for (lambda_index in seq_len(n_lambda)) {
        eta_out <- eta_all[
          ind.out,
          lambda_index
        ]

        loss <- 0.5 *
          (y_out - eta_out)^2

        lldiffFold[
          lambda_index,
          fold
        ] <- sum(loss)

        prevals[
          ind.out,
          lambda_index
        ] <- eta_out
      }
    } else if (identical(type, "logit")) {
      y_out <- y[ind.out]

      for (lambda_index in seq_len(n_lambda)) {
        eta_out <- eta_all[
          ind.out,
          lambda_index
        ]

        # 稳定计算：
        # -y * eta + log(1 + exp(eta))
        loss <- pmax(
          eta_out,
          0
        ) -
          y_out * eta_out +
          log1p(
            exp(
              -abs(eta_out)
            )
          )

        lldiffFold[
          lambda_index,
          fold
        ] <- sum(loss)

        prevals[
          ind.out,
          lambda_index
        ] <- plogis(
          eta_out
        )
      }
    } else {
      eta_train <- eta_all[
        ind.in,
        ,
        drop = FALSE
      ]

      for (lambda_index in seq_len(n_lambda)) {
        eta_current <- eta_all[,
          lambda_index
        ]

        full_loss <-
          .cvSGL_cox_negative_loglik(
            time = time,
            status = status,
            eta = eta_current
          )

        train_loss <-
          .cvSGL_cox_negative_loglik(
            time = time[ind.in],
            status = status[ind.in],
            eta = eta_train[,
              lambda_index
            ]
          )

        # 与 main 分支 coxCrossVal 的
        # full negative log-likelihood -
        # training negative log-likelihood
        # 保持一致。
        lldiffFold[
          lambda_index,
          fold
        ] <- full_loss - train_loss

        prevals[
          ind.out,
          lambda_index
        ] <- eta_current[ind.out]
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

  lldiff <- rowMeans(
    lldiffFold
  )

  llSD <- apply(
    lldiffFold,
    1L,
    stats::sd
  ) *
    sqrt(nfold)

  lambda_min_index <- which.min(
    lldiff
  )

  lambda_min <- lambda_path[
    lambda_min_index
  ]

  one_se_limit <- lldiff[
    lambda_min_index
  ] +
    llSD[
      lambda_min_index
    ]

  eligible <- which(
    lldiff <= one_se_limit
  )

  lambda_1se_index <- eligible[
    which.max(
      lambda_path[eligible]
    )
  ]

  lambda_1se <- lambda_path[
    lambda_1se_index
  ]

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

  class(result) <- "cv.SGL"

  result
}
