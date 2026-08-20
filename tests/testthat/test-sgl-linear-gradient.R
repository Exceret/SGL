test_that("linear gradient returns a single-column matrix", {
  set.seed(20240421)

  n <- 100L
  p <- 20L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  eta <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  y <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  actual <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(p, 1L))
  expect_true(all(is.finite(actual)))
})


test_that("linear gradient matches R reference", {
  set.seed(20240422)

  for (n in c(1L, 2L, 10L, 100L, 1000L, 10000L)) {
    for (p in c(1L, 2L, 5L, 20L, 100L)) {
      X <- matrix(
        rnorm(n * p),
        nrow = n,
        ncol = p
      )

      eta <- matrix(
        rnorm(n, sd = 10),
        nrow = n,
        ncol = 1L
      )

      y <- matrix(
        rnorm(n, sd = 10),
        nrow = n,
        ncol = 1L
      )

      expected <- crossprod(
        X,
        eta - y
      )

      actual <- sgl_linear_gradient_cpp(
        X = X,
        eta = eta,
        y = y
      )

      expect_true(is.matrix(actual))
      expect_identical(dim(actual), c(p, 1L))
      expect_true(all(is.finite(actual)))

      expect_lt(
        max(abs(actual - expected)),
        1e-8
      )
    }
  }
})


test_that("linear gradient is zero for zero residuals", {
  set.seed(20240423)

  n <- 100L
  p <- 20L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  y <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  eta <- y

  actual <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expected <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(p, 1L))
  expect_lt(
    max(abs(actual - expected)),
    1e-12
  )
})


test_that("zero design matrix produces zero gradient", {
  n <- 100L
  p <- 20L

  X <- matrix(
    0,
    nrow = n,
    ncol = p
  )

  eta <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  y <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  actual <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expected <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  expect_identical(actual, expected)
})


test_that("linear gradient changes sign with residual", {
  set.seed(20240424)

  n <- 100L
  p <- 10L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  eta <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  y <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  gradient_1 <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  gradient_2 <- sgl_linear_gradient_cpp(
    X = X,
    eta = y,
    y = eta
  )

  expect_lt(
    max(abs(gradient_1 + gradient_2)),
    1e-10
  )
})


test_that("linear gradient is consistent with finite differences", {
  set.seed(20240425)

  n <- 40L
  p <- 6L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- matrix(
    rnorm(p),
    nrow = p,
    ncol = 1L
  )

  intercept <- 0.35

  y <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  eta <- matrix(
    intercept + X %*% beta,
    nrow = n,
    ncol = 1L
  )

  analytic <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  objective <- function(beta_value) {
    eta_value <- matrix(
      intercept + X %*% beta_value,
      nrow = n,
      ncol = 1L
    )

    0.5 * sum((eta_value - y)^2)
  }

  h <- 1e-6

  numeric_gradient <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  for (j in seq_len(p)) {
    beta_plus <- beta
    beta_minus <- beta

    beta_plus[j, 1L] <-
      beta_plus[j, 1L] + h

    beta_minus[j, 1L] <-
      beta_minus[j, 1L] - h

    numeric_gradient[j, 1L] <- (objective(beta_plus) -
      objective(beta_minus)) /
      (2 * h)
  }

  expect_lt(
    max(abs(analytic - numeric_gradient)),
    1e-6
  )
})


test_that("linear gradient handles large finite values", {
  set.seed(20240426)

  n <- 100L
  p <- 10L

  X <- matrix(
    rnorm(n * p, sd = 1e3),
    nrow = n,
    ncol = p
  )

  eta <- matrix(
    rnorm(n, sd = 1e3),
    nrow = n,
    ncol = 1L
  )

  y <- matrix(
    rnorm(n, sd = 1e3),
    nrow = n,
    ncol = 1L
  )

  actual <- sgl_linear_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expected <- crossprod(
    X,
    eta - y
  )

  expect_true(all(is.finite(actual)))
  expect_lt(
    max(abs(actual - expected)),
    1e-7 # bigger a little than 1e-8
  )
})


test_that("invalid matrix shapes are rejected", {
  X <- matrix(
    1,
    nrow = 3L,
    ncol = 2L
  )

  eta_vector <- c(0, 1, -1)
  y <- matrix(
    c(0, 1, 0),
    ncol = 1L
  )

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta_vector,
      y = y
    ),
    "Not a matrix"
  )

  eta <- matrix(
    c(0, 1, -1),
    ncol = 1L
  )

  y_wrong_n <- matrix(
    c(0, 1),
    ncol = 1L
  )

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta,
      y = y_wrong_n
    ),
    "must equal nrow"
  )

  eta_wrong_n <- matrix(
    c(0, 1),
    ncol = 1L
  )

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta_wrong_n,
      y = y
    ),
    "must equal nrow"
  )
})


test_that("non-finite values are rejected", {
  X <- matrix(
    1,
    nrow = 3L,
    ncol = 2L
  )

  eta <- matrix(
    c(-1, 0, 1),
    ncol = 1L
  )

  y <- matrix(
    c(0, 1, 0),
    ncol = 1L
  )

  X[1L, 1L] <- NA_real_

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta,
      y = y
    ),
    "finite"
  )

  X[1L, 1L] <- 1
  eta[2L, 1L] <- Inf

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta,
      y = y
    ),
    "finite"
  )

  eta[2L, 1L] <- 0
  y[3L, 1L] <- NaN

  expect_error(
    sgl_linear_gradient_cpp(
      X = X,
      eta = eta,
      y = y
    ),
    "finite"
  )
})
