test_that("linear intercept gradient matches R reference", {
  set.seed(20241001)

  for (n in c(0L, 1L, 2L, 10L, 100L, 1000L)) {
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

    expected <- sum(eta - y)

    actual <- sgl_linear_intercept_gradient_cpp(
      eta = eta,
      y = y
    )

    expect_type(actual, "double")
    expect_length(actual, 1L)
    expect_lt(
      abs(actual - expected),
      1e-10
    )
  }
})


test_that("Logistic intercept gradient matches R reference", {
  set.seed(20241002)

  for (n in c(1L, 2L, 10L, 100L, 1000L, 10000L)) {
    eta <- matrix(
      rnorm(n, sd = 10),
      nrow = n,
      ncol = 1L
    )

    y <- matrix(
      rbinom(
        n,
        size = 1L,
        prob = 0.5
      ),
      nrow = n,
      ncol = 1L
    )

    expected <- sum(
      plogis(eta) - y
    )

    actual <- sgl_logistic_intercept_gradient_cpp(
      eta = eta,
      y = y
    )

    expect_true(is.finite(actual))
    expect_lt(
      abs(actual - expected),
      1e-10
    )
  }
})


test_that("Logistic intercept gradient is stable for extreme eta", {
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

  expected <- sum(
    plogis(eta) - y
  )

  actual <- sgl_logistic_intercept_gradient_cpp(
    eta = eta,
    y = y
  )

  expect_true(is.finite(actual))
  expect_lt(
    abs(actual - expected),
    1e-12
  )
})


test_that("Logistic intercept Hessian matches R reference", {
  set.seed(20241003)

  for (n in c(0L, 1L, 2L, 10L, 100L, 1000L)) {
    eta <- matrix(
      rnorm(n, sd = 10),
      nrow = n,
      ncol = 1L
    )

    expected <- sum(
      plogis(eta) *
        (1 - plogis(eta))
    )

    actual <- sgl_logistic_intercept_hessian_cpp(
      eta = eta
    )

    expect_type(actual, "double")
    expect_length(actual, 1L)
    expect_true(is.finite(actual))
    expect_gte(actual, 0)
    expect_lt(
      abs(actual - expected),
      1e-12
    )
  }
})


test_that("Logistic intercept Hessian is stable for extreme eta", {
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

  actual <- sgl_logistic_intercept_hessian_cpp(
    eta
  )

  expect_true(is.finite(actual))
  expect_gte(actual, 0)
  expect_lte(actual, 2.5)
})


test_that("intercept and eta update preserve matrix formats", {
  intercept <- matrix(
    0.5,
    nrow = 1L,
    ncol = 1L
  )

  eta <- matrix(
    c(-2, -1, 0, 1, 2),
    ncol = 1L
  )

  actual <- sgl_update_intercept_eta_cpp(
    intercept = intercept,
    eta = eta,
    delta_intercept = 0.75
  )

  expect_true(is.list(actual))
  expect_true(is.matrix(actual$intercept))
  expect_true(is.matrix(actual$eta))

  expect_identical(
    dim(actual$intercept),
    c(1L, 1L)
  )

  expect_identical(
    dim(actual$eta),
    c(5L, 1L)
  )
})


test_that("intercept and eta update matches reference", {
  set.seed(20241004)

  for (n in c(0L, 1L, 10L, 100L, 1000L)) {
    intercept <- matrix(
      rnorm(1L),
      nrow = 1L,
      ncol = 1L
    )

    eta <- matrix(
      rnorm(n),
      nrow = n,
      ncol = 1L
    )

    delta_intercept <- rnorm(1L)

    expected_intercept <-
      intercept + delta_intercept

    expected_eta <-
      eta + delta_intercept

    actual <- sgl_update_intercept_eta_cpp(
      intercept = intercept,
      eta = eta,
      delta_intercept = delta_intercept
    )

    expect_lt(
      max(abs(
        actual$intercept -
          expected_intercept
      )),
      1e-12
    )

    if (n == 0L) {
      expect_true(is.matrix(actual$eta))
      expect_identical(
        dim(actual$eta),
        c(0L, 1L)
      )
      expect_length(actual$eta, 0L)
    } else {
      expect_lt(
        max(abs(
          actual$eta -
            expected_eta
        )),
        1e-12
      )
    }
  }
})


test_that("zero intercept update preserves exact values", {
  set.seed(20241005)

  intercept <- matrix(
    rnorm(1L),
    nrow = 1L,
    ncol = 1L
  )

  eta <- matrix(
    rnorm(100L),
    nrow = 100L,
    ncol = 1L
  )

  actual <- sgl_update_intercept_eta_cpp(
    intercept = intercept,
    eta = eta,
    delta_intercept = 0
  )

  expect_identical(
    actual$intercept,
    intercept
  )

  expect_identical(
    actual$eta,
    eta
  )
})


test_that("repeated intercept updates equal one combined update", {
  set.seed(20241006)

  intercept <- matrix(
    rnorm(1L),
    nrow = 1L,
    ncol = 1L
  )

  eta <- matrix(
    rnorm(100L),
    nrow = 100L,
    ncol = 1L
  )

  delta_1 <- 0.25
  delta_2 <- -0.75

  first <- sgl_update_intercept_eta_cpp(
    intercept = intercept,
    eta = eta,
    delta_intercept = delta_1
  )

  second <- sgl_update_intercept_eta_cpp(
    intercept = first$intercept,
    eta = first$eta,
    delta_intercept = delta_2
  )

  combined <- sgl_update_intercept_eta_cpp(
    intercept = intercept,
    eta = eta,
    delta_intercept = delta_1 + delta_2
  )

  expect_lt(
    max(abs(
      second$intercept -
        combined$intercept
    )),
    1e-12
  )

  expect_lt(
    max(abs(
      second$eta -
        combined$eta
    )),
    1e-12
  )
})


test_that("intercept update rejects invalid shapes", {
  expect_error(
    sgl_update_intercept_eta_cpp(
      intercept = matrix(
        c(0, 1),
        nrow = 2L,
        ncol = 1L
      ),
      eta = matrix(0, nrow = 3L, ncol = 1L),
      delta_intercept = 0.1
    ),
    "1 x 1"
  )

  expect_error(
    sgl_update_intercept_eta_cpp(
      intercept = matrix(
        0,
        nrow = 1L,
        ncol = 1L
      ),
      eta = matrix(
        0,
        nrow = 1L,
        ncol = 2L
      ),
      delta_intercept = 0.1
    ),
    "single-column matrix"
  )

  expect_error(
    sgl_linear_intercept_gradient_cpp(
      eta = matrix(0, nrow = 3L, ncol = 1L),
      y = matrix(0, nrow = 2L, ncol = 1L)
    ),
    "nrow"
  )
})


test_that("Logistic intercept functions reject non-binary response", {
  eta <- matrix(
    c(-1, 0, 1),
    ncol = 1L
  )

  expect_error(
    sgl_logistic_intercept_gradient_cpp(
      eta = eta,
      y = matrix(
        c(0, 0.5, 1),
        ncol = 1L
      )
    ),
    "0 and 1"
  )

  expect_error(
    sgl_logistic_intercept_gradient_cpp(
      eta = eta,
      y = matrix(
        c(0, 1, 2),
        ncol = 1L
      )
    ),
    "0 and 1"
  )
})


test_that("intercept functions reject non-finite values", {
  eta <- matrix(
    c(-1, 0, 1),
    ncol = 1L
  )

  y <- matrix(
    c(0, 1, 0),
    ncol = 1L
  )

  eta[2L, 1L] <- NA_real_

  expect_error(
    sgl_logistic_intercept_gradient_cpp(
      eta = eta,
      y = y
    ),
    "finite"
  )

  eta[2L, 1L] <- Inf

  expect_error(
    sgl_logistic_intercept_hessian_cpp(
      eta = eta
    ),
    "finite"
  )

  eta[2L, 1L] <- 0
  y[3L, 1L] <- NaN

  expect_error(
    sgl_linear_intercept_gradient_cpp(
      eta = eta,
      y = y
    ),
    "finite"
  )
})
