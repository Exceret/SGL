make_sgl_test_data <- function(
  n = 120L,
  p = 12L,
  n_groups = 6L,
  seed = 20260820
) {
  set.seed(seed)

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  index <- rep(
    seq_len(n_groups),
    length.out = p
  )

  beta_true <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  active <- seq_len(
    min(4L, p)
  )

  beta_true[active, 1L] <-
    c(0.8, -0.6, 0.4, -0.3)[
      seq_along(active)
    ]

  linear_predictor <- as.numeric(
    X %*% beta_true
  )

  y_linear <- 0.5 +
    linear_predictor +
    rnorm(
      n,
      sd = 0.5
    )

  beta_logit <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  active_logit <- seq_len(
    min(4L, p)
  )

  beta_logit[active_logit, 1L] <-
    c(0.25, -0.20, 0.15, -0.10)[
      seq_along(active_logit)
    ]

  eta_logit <- as.numeric(
    X %*% beta_logit
  )

  probability <- plogis(
    -0.10 + eta_logit
  )

  y_logit <- rbinom(
    n,
    size = 1L,
    prob = probability
  )

  if (
    sum(y_logit) == 0L ||
      sum(y_logit) == n
  ) {
    y_logit <- rbinom(
      n,
      size = 1L,
      prob = 0.5
    )
  }

  time <- rexp(
    n,
    rate = 0.2
  )

  status <- rbinom(
    n,
    size = 1L,
    prob = 0.6
  )

  if (!any(status == 1L)) {
    status[1L] <- 1L
  }

  list(
    X = X,
    index = index,
    y_linear = y_linear,
    y_logit = y_logit,
    y_cox = cbind(
      time = time,
      status = status
    )
  )
}

test_that("SGL dispatches all supported model types", {
  data <- make_sgl_test_data(
    n = 80L,
    p = 10L,
    n_groups = 5L
  )

  linear_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6
  )

  logit_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6,
    step = 0.05
  )

  cox_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_cox
    ),
    index = data$index,
    type = "cox",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6
  )

  expect_identical(
    linear_fit$type,
    "linear"
  )

  expect_identical(
    logit_fit$type,
    "logit"
  )

  expect_identical(
    cox_fit$type,
    "cox"
  )

  expect_true(
    all(is.finite(linear_fit$beta))
  )

  expect_true(
    all(is.finite(logit_fit$beta))
  )

  expect_true(
    all(is.finite(cox_fit$beta))
  )

  linear_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6
  )

  logit_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6,
    step = 0.01
  )

  cox_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      time = data$y_cox[, 1L],
      status = data$y_cox[, 2L]
    ),
    index = data$index,
    type = "cox",
    lambdas = c(0.1, 0.05),
    maxit = 800L,
    thresh = 1e-6
  )
})
