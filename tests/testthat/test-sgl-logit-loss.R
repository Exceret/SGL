reference_logistic_nll <- function(eta, y) {
  stopifnot(
    is.matrix(eta),
    ncol(eta) == 1L,
    is.matrix(y),
    ncol(y) == 1L
  )

  sum(
    pmax(eta, 0) +
      log1p(exp(-abs(eta))) -
      y * eta
  )
}


test_that("logistic negative log-likelihood returns a scalar", {
  eta <- matrix(c(-2, -1, 0, 1, 2), ncol = 1L)
  y <- matrix(c(0, 0, 1, 1, 1), ncol = 1L)

  actual <- sgl_logistic_nll_cpp(eta, y)

  expect_type(actual, "double")
  expect_length(actual, 1L)
  expect_true(is.finite(actual))
})


test_that("logistic negative log-likelihood matches stable R reference", {
  set.seed(20240331)

  for (n in c(1L, 2L, 10L, 100L, 1000L, 10000L)) {
    for (iter in seq_len(50L)) {
      eta <- matrix(
        rnorm(n, mean = 0, sd = 10),
        nrow = n,
        ncol = 1L
      )

      y <- matrix(
        rbinom(n, size = 1L, prob = 0.5),
        nrow = n,
        ncol = 1L
      )

      expected <- reference_logistic_nll(eta, y)
      actual <- sgl_logistic_nll_cpp(eta, y)

      expect_true(is.finite(actual))
      expect_lt(
        abs(actual - expected),
        1e-8
      )
    }
  }
})


test_that("logistic negative log-likelihood is stable for extreme predictors", {
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

  expected <- reference_logistic_nll(eta, y)
  actual <- sgl_logistic_nll_cpp(eta, y)

  expect_true(is.finite(actual))
  expect_lt(
    abs(actual - expected),
    1e-8
  )
})


test_that("zero loss is obtained for perfectly classified finite values", {
  eta <- matrix(
    c(-100, -50, 50, 100),
    ncol = 1L
  )

  y <- matrix(
    c(0, 0, 1, 1),
    ncol = 1L
  )

  actual <- sgl_logistic_nll_cpp(eta, y)

  expect_true(is.finite(actual))
  expect_gte(actual, 0)
  expect_lt(actual, 1e-12)
})


test_that("loss is non-negative", {
  set.seed(20240401)

  eta <- matrix(rnorm(1000L, sd = 20), ncol = 1L)
  y <- matrix(rbinom(1000L, 1L, 0.5), ncol = 1L)

  actual <- sgl_logistic_nll_cpp(eta, y)

  expect_true(is.finite(actual))
  expect_gte(actual, 0)
})


test_that("loss is additive over observations", {
  set.seed(20240402)

  n <- 200L

  eta <- matrix(rnorm(n, sd = 5), ncol = 1L)
  y <- matrix(rbinom(n, 1L, 0.5), ncol = 1L)

  full <- sgl_logistic_nll_cpp(eta, y)

  first <- sgl_logistic_nll_cpp(
    eta[1:100, , drop = FALSE],
    y[1:100, , drop = FALSE]
  )

  second <- sgl_logistic_nll_cpp(
    eta[101:200, , drop = FALSE],
    y[101:200, , drop = FALSE]
  )

  expect_lt(
    abs(full - (first + second)),
    1e-8
  )
})


test_that("single-column matrix inputs are required", {
  eta <- matrix(rnorm(6L), nrow = 2L, ncol = 3L)
  y <- matrix(c(0, 1), ncol = 1L)

  expect_error(
    sgl_logistic_nll_cpp(eta, y),
    "single-column matrix"
  )
})


test_that("binary response is required", {
  eta <- matrix(c(-1, 0, 1), ncol = 1L)

  expect_error(
    sgl_logistic_nll_cpp(
      eta,
      matrix(c(0, 0.5, 1), ncol = 1L)
    ),
    "0 and 1"
  )

  expect_error(
    sgl_logistic_nll_cpp(
      eta,
      matrix(c(0, 1, 2), ncol = 1L)
    ),
    "0 and 1"
  )
})


test_that("non-finite values are rejected", {
  y <- matrix(c(0, 1, 0), ncol = 1L)

  eta_na <- matrix(c(-1, NA_real_, 1), ncol = 1L)
  expect_error(
    sgl_logistic_nll_cpp(eta_na, y),
    "finite"
  )

  eta_inf <- matrix(c(-1, Inf, 1), ncol = 1L)
  expect_error(
    sgl_logistic_nll_cpp(eta_inf, y),
    "finite"
  )

  y_na <- matrix(c(0, NA_real_, 1), ncol = 1L)
  eta <- matrix(c(-1, 0, 1), ncol = 1L)

  expect_error(
    sgl_logistic_nll_cpp(eta, y_na),
    "finite"
  )
})
