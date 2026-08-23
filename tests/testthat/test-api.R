skip()
skip_if_not_installed("SGL")

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

  linear_predictor <- drop(
    X %*% beta_true
  )

  y_linear <- 0.5 +
    linear_predictor +
    rnorm(
      n,
      sd = 0.5
    )

  # --------------------------------------------------
  # Logistic 数据
  # --------------------------------------------------

  beta_logit <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  active_logit <- seq_len(
    min(4L, p)
  )

  # 原来的系数过小，导致 Logistic 信号较弱。
  # 这里提高到中等强度，避免数据过于接近随机分类。
  beta_logit[active_logit, 1L] <-
    c(0.8, -0.6, 0.45, -0.35)[
      seq_along(active_logit)
    ]

  eta_logit <- drop(
    X %*% beta_logit
  )

  # 目标阳性率。
  # 通过校准截距，而不是固定使用 -0.10，
  # 使平均生成概率接近 target_prevalence。
  target_prevalence <- 0.5

  intercept_logit <- uniroot(
    f = function(intercept) {
      mean(
        plogis(
          intercept + eta_logit
        )
      ) -
        target_prevalence
    },
    interval = c(-20, 20),
    tol = 1e-10
  )$root

  probability <- plogis(
    intercept_logit + eta_logit
  )

  y_logit <- rbinom(
    n,
    size = 1L,
    prob = probability
  )

  # 极低概率的保护措施：
  # 不再用 Bernoulli(0.5) 覆盖整个响应，
  # 而只修正一个最合理的样本，保留原始信号。
  if (!any(y_logit == 1L)) {
    y_logit[
      which.max(probability)
    ] <- 1L
  }

  if (!any(y_logit == 0L)) {
    y_logit[
      which.min(probability)
    ] <- 0L
  }

  # --------------------------------------------------
  # Cox 数据
  # --------------------------------------------------

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

is_tolerant <- function(x, y, tol = 1e-8) {
  diff <- max(abs(x - y))
  flag <- diff < tol
  if (!flag) {
    message("Tolerant: ", tol, ", diff: ", diff)
  }
  flag
}

test_that("SGL dispatches supported linear model", {
  data <- make_sgl_test_data()

  linear_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear"
  )

  linear_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear"
  )

  expect_true(is_tolerant(linear_fit$beta, linear_fit_raw$beta))
  expect_true(is_tolerant(linear_fit$intercept, linear_fit_raw$intercept))
  expect_true(is_tolerant(linear_fit$lambdas, linear_fit_raw$lambdas))
  expect_true(is_tolerant(
    linear_fit$X.transform$X.means,
    linear_fit_raw$X.transform$X.means
  ))
  expect_true(is_tolerant(
    linear_fit$X.transform$X.scale,
    linear_fit_raw$X.transform$X.scale
  ))

  cv_linear <- cvSGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = linear_fit$lambdas
  )
  # cv_linear2 <- SGL::cvSGL(
  #   data = list(
  #     x = data$X,
  #     y = data$y_linear
  #   ),
  #   index = data$index,
  #   type = "linear",
  #   lambdas = linear_fit$lambdas
  # )
  cv_linear_raw <- SGL::cvSGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = linear_fit_raw$lambdas
  )
})

# ----------------------------------------------------------------------------
test_that("SGL dispatches supported logistic model", {
  data <- make_sgl_test_data()

  logit_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    step = 0.01
  )

  logit_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    step = 0.01
  )

  expect_true(is_tolerant(logit_fit$beta, logit_fit_raw$beta))
  expect_true(is_tolerant(logit_fit$intercept, logit_fit_raw$intercept))
  expect_true(is_tolerant(logit_fit$lambdas, logit_fit_raw$lambdas))
  expect_true(is_tolerant(
    logit_fit$X.transform$X.means,
    logit_fit_raw$X.transform$X.means
  ))
  expect_true(is_tolerant(
    logit_fit$X.transform$X.scale,
    logit_fit_raw$X.transform$X.scale
  ))

  cv_logit <- cvSGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    lambdas = logit_fit$lambdas
  )
  cv_logit_raw <- SGL::cvSGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    lambdas = logit_fit_raw$lambdas
  )

  expect_s3_class(cv_logit, "cvSGL")
  expect_s3_class(cv_logit_raw, "cvSGL")
})

# ----------------------------------------------------------------------------
test_that("SGL dispatches supported cox model", {
  data <- make_sgl_test_data(n = 50L)

  cox_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_cox
    ),
    index = data$index,
    type = "cox",
    maxit = 800L,
    thresh = 1e-6
  )

  cox_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      time = data$y_cox[, 1L],
      status = data$y_cox[, 2L]
    ),
    index = data$index,
    type = "cox",
    maxit = 800L,
    thresh = 1e-6
  )

  expect_true(is_tolerant(cox_fit$beta, cox_fit_raw$beta))
  expect_true(is_tolerant(cox_fit$lambdas, cox_fit_raw$lambdas))
  expect_true(is_tolerant(
    cox_fit$X.transform$X.means,
    cox_fit_raw$X.transform$X.means
  ))
  expect_true(is_tolerant(
    cox_fit$X.transform$X.scale,
    cox_fit_raw$X.transform$X.scale
  ))

  cv_cox <- cvSGL(
    data = list(
      x = data$X,
      y = data$y_cox
    ),
    index = data$index,
    type = "cox",
    lambdas = cox_fit$lambdas
  )
  cv_cox_raw <- SGL::cvSGL(
    data = list(
      x = data$X,
      time = data$y_cox[, 1L],
      status = data$y_cox[, 2L]
    ),
    index = data$index,
    type = "cox",
    lambdas = cox_fit_raw$lambdas
  )

  expect_s3_class(cv_cox, "cvSGL")
  expect_s3_class(cv_cox_raw, "cvSGL")
})

# microbenchmark::microbenchmark(
#   cpp = {
#     linear_fit <- SGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear"
#     )

#     cv_linear <- cvSGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear",
#       lambdas = linear_fit$lambdas
#     )
#   },
#   r = {
#     linear_fit_raw <- SGL::SGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear"
#     )
#     cv_linear_raw <- SGL::cvSGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear",
#       lambdas = linear_fit_raw$lambdas
#     )
#   }
# )

# peakRAM::peakRAM(
#   cpp = {
#     linear_fit <- SGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear"
#     )

#     cv_linear <- cvSGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear",
#       lambdas = linear_fit$lambdas
#     )
#   },
#   r = {
#     linear_fit_raw <- SGL::SGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear"
#     )
#     cv_linear_raw <- SGL::cvSGL(
#       data = list(
#         x = data$X,
#         y = data$y_linear
#       ),
#       index = data$index,
#       type = "linear",
#       lambdas = linear_fit_raw$lambdas
#     )
#   }
# )
