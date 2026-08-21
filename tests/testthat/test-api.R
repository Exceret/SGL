skip()

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

test_that("SGL dispatches all supported model types", {
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

  cv_linear <- cvSGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = linear_fit$lambdas
  )
  cv_linear2 <- SGL::cvSGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = linear_fit$lambdas
  )
  cv_linear_raw <- SGL::cvSGL(
    data = list(
      x = data$X,
      y = data$y_linear
    ),
    index = data$index,
    type = "linear",
    lambdas = linear_fit_raw$lambdas
  )

  # ----------------------------------------------------------------------------
  logit_fit <- SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
    step = 0.01
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

  logit_fit_raw <- SGL::SGL(
    data = list(
      x = data$X,
      y = data$y_logit
    ),
    index = data$index,
    type = "logit",
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
