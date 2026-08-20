reference_logistic_nll_gradient <- function(X, eta, y) {
  stopifnot(
    is.matrix(X),
    is.matrix(eta),
    ncol(eta) == 1L,
    is.matrix(y),
    ncol(y) == 1L
  )

  crossprod(
    X,
    plogis(eta) - y
  )
}


test_that("Logistic gradient returns a single-column matrix", {
  set.seed(20240411)

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
    rbinom(n, size = 1L, prob = 0.5),
    nrow = n,
    ncol = 1L
  )

  actual <- sgl_logistic_nll_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(p, 1L))
  expect_true(all(is.finite(actual)))
})


test_that("Logistic gradient matches R reference", {
  set.seed(20240412)

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
        rbinom(n, size = 1L, prob = 0.5),
        nrow = n,
        ncol = 1L
      )

      expected <- reference_logistic_nll_gradient(
        X = X,
        eta = eta,
        y = y
      )

      actual <- sgl_logistic_nll_gradient_cpp(
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


test_that("Logistic gradient is stable for extreme predictors", {
  set.seed(20240413)

  X <- matrix(
    rnorm(9L * 5L),
    nrow = 9L,
    ncol = 5L
  )

  eta <- matrix(
    c(
      -10000,
      -1000,
      -745,
      -100,
      0,
      100,
      745,
      1000,
      10000
    ),
    ncol = 1L
  )

  y <- matrix(
    c(0, 1, 0, 1, 0, 1, 0, 1, 0),
    ncol = 1L
  )

  expected <- reference_logistic_nll_gradient(
    X = X,
    eta = eta,
    y = y
  )

  actual <- sgl_logistic_nll_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(5L, 1L))
  expect_true(all(is.finite(actual)))

  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("zero design matrix produces zero gradient", {
  X <- matrix(
    0,
    nrow = 100L,
    ncol = 20L
  )

  eta <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  y <- matrix(
    rbinom(100L, size = 1L, prob = 0.5),
    ncol = 1L
  )

  actual <- sgl_logistic_nll_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expected <- matrix(
    0,
    nrow = 20L,
    ncol = 1L
  )

  expect_identical(actual, expected)
})


test_that("all-zero response is handled", {
  set.seed(20240414)

  X <- matrix(
    rnorm(100L * 10L),
    nrow = 100L,
    ncol = 10L
  )

  eta <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  y <- matrix(
    0,
    nrow = 100L,
    ncol = 1L
  )

  expected <- crossprod(
    X,
    plogis(eta)
  )

  actual <- sgl_logistic_nll_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("all-one response is handled", {
  set.seed(20240415)

  X <- matrix(
    rnorm(100L * 10L),
    nrow = 100L,
    ncol = 10L
  )

  eta <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  y <- matrix(
    1,
    nrow = 100L,
    ncol = 1L
  )

  expected <- crossprod(
    X,
    plogis(eta) - 1
  )

  actual <- sgl_logistic_nll_gradient_cpp(
    X = X,
    eta = eta,
    y = y
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("gradient is consistent with finite differences", {
  set.seed(20240416)

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

  eta <- matrix(
    intercept + X %*% beta,
    nrow = n,
    ncol = 1L
  )

  y <- matrix(
    rbinom(n, size = 1L, prob = 0.5),
    nrow = n,
    ncol = 1L
  )

  analytic <- sgl_logistic_nll_gradient_cpp(
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

    sum(
      pmax(eta_value, 0) +
        log1p(exp(-abs(eta_value))) -
        y * eta_value
    )
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

    beta_plus[j, 1L] <- beta_plus[j, 1L] + h
    beta_minus[j, 1L] <- beta_minus[j, 1L] - h

    numeric_gradient[j, 1L] <- (objective(beta_plus) -
      objective(beta_minus)) /
      (2 * h)
  }

  expect_lt(
    max(abs(analytic - numeric_gradient)),
    1e-6
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
    sgl_logistic_nll_gradient_cpp(
      X,
      eta_vector,
      y
    ),
    "Not a matrix"
  )

  eta <- matrix(
    c(0, 1, -1),
    ncol = 1L
  )

  y_wrong_p <- matrix(
    c(0, 1, 0, 1),
    ncol = 2L
  )

  expect_error(
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      y_wrong_p
    ),
    "single-column matrix"
  )
})


test_that("non-binary responses are rejected", {
  X <- matrix(
    1,
    nrow = 3L,
    ncol = 2L
  )

  eta <- matrix(
    c(-1, 0, 1),
    ncol = 1L
  )

  expect_error(
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      matrix(c(0, 0.5, 1), ncol = 1L)
    ),
    "0 and 1"
  )

  expect_error(
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      matrix(c(0, 1, 2), ncol = 1L)
    ),
    "0 and 1"
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
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      y
    ),
    "finite"
  )

  X[1L, 1L] <- 1
  eta[2L, 1L] <- Inf

  expect_error(
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      y
    ),
    "finite"
  )

  eta[2L, 1L] <- 0
  y[3L, 1L] <- NA_real_

  expect_error(
    sgl_logistic_nll_gradient_cpp(
      X,
      eta,
      y
    ),
    "finite"
  )
})
