test_that("logistic probability preserves single-column matrix format", {
  eta <- matrix(
    c(-3, -1, 0, 1, 3),
    ncol = 1L
  )

  actual <- sgl_logistic_probability_cpp(eta)

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), dim(eta))
})


test_that("logistic probability matches plogis", {
  set.seed(20240321)

  for (n in c(1L, 2L, 10L, 100L, 1000L, 10000L)) {
    eta <- matrix(
      rnorm(n, mean = 0, sd = 10),
      nrow = n,
      ncol = 1L
    )

    expected <- plogis(eta)
    actual <- sgl_logistic_probability_cpp(eta)

    expect_true(is.matrix(actual))
    expect_identical(dim(actual), c(n, 1L))
    expect_true(all(is.finite(actual)))
    expect_true(all(actual >= 0))
    expect_true(all(actual <= 1))

    expect_lt(
      max(abs(actual - expected)),
      1e-12
    )
  }
})


test_that("logistic probability is stable for extreme predictors", {
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

  actual <- sgl_logistic_probability_cpp(eta)
  expected <- plogis(eta)

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(9L, 1L))
  expect_true(all(is.finite(actual)))
  expect_lt(max(abs(actual - expected)), 1e-12)
})


test_that("logistic probability has correct symmetry", {
  set.seed(20240322)

  eta <- matrix(
    rnorm(1000L, mean = 0, sd = 20),
    ncol = 1L
  )

  p_eta <- sgl_logistic_probability_cpp(eta)
  p_neg_eta <- sgl_logistic_probability_cpp(-eta)

  expect_lt(
    max(abs(p_eta + p_neg_eta - 1)),
    1e-12
  )
})


test_that("logistic probability is monotone", {
  eta <- matrix(
    seq(-100, 100, length.out = 1001L),
    ncol = 1L
  )

  actual <- sgl_logistic_probability_cpp(eta)

  expect_true(all(diff(actual[, 1L]) >= 0))
})


test_that("logistic probability handles zero predictor", {
  eta <- matrix(0, nrow = 100L, ncol = 1L)

  actual <- sgl_logistic_probability_cpp(eta)

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(100L, 1L))
  expect_equal(
    actual,
    matrix(0.5, nrow = 100L, ncol = 1L),
    tolerance = 0
  )
})


test_that("logistic probability rejects non-column matrices", {
  eta <- matrix(
    rnorm(6L),
    nrow = 2L,
    ncol = 3L
  )

  expect_error(
    sgl_logistic_probability_cpp(eta),
    "single-column matrix"
  )
})


test_that("logistic probability rejects non-finite values", {
  eta <- matrix(
    c(0, NA_real_, 1),
    ncol = 1L
  )

  expect_error(
    sgl_logistic_probability_cpp(eta),
    "finite"
  )

  eta[2L, 1L] <- NaN

  expect_error(
    sgl_logistic_probability_cpp(eta),
    "finite"
  )

  eta[2L, 1L] <- Inf

  expect_error(
    sgl_logistic_probability_cpp(eta),
    "finite"
  )
})
