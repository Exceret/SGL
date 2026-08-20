test_that("eta update returns a single-column matrix", {
  set.seed(20240301)

  n <- 100L
  p <- 20L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- matrix(rnorm(n), ncol = 1L)
  delta_beta <- matrix(rnorm(p), ncol = 1L)

  actual <- sgl_update_eta_cpp(
    X = X,
    eta = eta,
    delta_beta = delta_beta
  )

  expected <- eta + X %*% delta_beta

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(n, 1L))
  expect_lt(max(abs(actual - expected)), 1e-8)
})


test_that("group eta update returns a single-column matrix", {
  set.seed(20240302)

  n <- 100L
  p <- 20L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- matrix(rnorm(n), ncol = 1L)
  delta_beta <- matrix(rnorm(p), ncol = 1L)

  first_col <- 5L
  last_col <- 12L
  selected <- seq.int(first_col + 1L, last_col + 1L)

  expected <- eta +
    X[, selected, drop = FALSE] %*%
      delta_beta[selected, , drop = FALSE]

  actual <- sgl_update_eta_group_cpp(
    X = X,
    eta = eta,
    delta_beta = delta_beta,
    first_col = first_col,
    last_col = last_col
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(n, 1L))
  expect_lt(max(abs(actual - expected)), 1e-8)
})


test_that("full eta update equals grouped eta updates", {
  set.seed(20240303)

  n <- 300L
  p <- 40L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- matrix(rnorm(n), ncol = 1L)
  delta_beta <- matrix(rnorm(p), ncol = 1L)

  full <- sgl_update_eta_cpp(
    X = X,
    eta = eta,
    delta_beta = delta_beta
  )

  grouped <- sgl_update_eta_group_cpp(
    X = X,
    eta = eta,
    delta_beta = delta_beta,
    first_col = 0L,
    last_col = 9L
  )

  grouped <- sgl_update_eta_group_cpp(
    X = X,
    eta = grouped,
    delta_beta = delta_beta,
    first_col = 10L,
    last_col = 19L
  )

  grouped <- sgl_update_eta_group_cpp(
    X = X,
    eta = grouped,
    delta_beta = delta_beta,
    first_col = 20L,
    last_col = 39L
  )

  expect_true(is.matrix(grouped))
  expect_identical(dim(grouped), c(n, 1L))
  expect_lt(max(abs(full - grouped)), 1e-8)
})


test_that("eta update handles zero coefficient increments", {
  set.seed(20240304)

  n <- 200L
  p <- 50L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- matrix(rnorm(n), ncol = 1L)
  delta_beta <- matrix(0, nrow = p, ncol = 1L)

  actual <- sgl_update_eta_cpp(
    X = X,
    eta = eta,
    delta_beta = delta_beta
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(n, 1L))
  expect_identical(actual, eta)
})


test_that("gradient returns a single-column matrix", {
  set.seed(20240305)

  n <- 100L
  p <- 20L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  residual <- matrix(rnorm(n), ncol = 1L)

  actual <- sgl_gradient_xt_residual_cpp(
    X = X,
    residual = residual
  )

  expected <- crossprod(X, residual)

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(p, 1L))
  expect_lt(max(abs(actual - expected)), 1e-8)
})


test_that("group gradient returns a single-column matrix", {
  set.seed(20240306)

  n <- 100L
  p <- 20L

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  residual <- matrix(rnorm(n), ncol = 1L)

  first_col <- 5L
  last_col <- 12L
  selected <- seq.int(first_col + 1L, last_col + 1L)

  expected <- matrix(0, nrow = p, ncol = 1L)
  expected[selected, 1L] <- crossprod(
    X[, selected, drop = FALSE],
    residual
  )

  actual <- sgl_gradient_group_xt_residual_cpp(
    X = X,
    residual = residual,
    first_col = first_col,
    last_col = last_col
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(p, 1L))
  expect_lt(max(abs(actual - expected)), 1e-8)
})


test_that("dimensions and finite values are validated", {
  X <- matrix(1, nrow = 3L, ncol = 2L)

  expect_error(
    sgl_update_eta_cpp(
      X = X,
      eta = matrix(1, nrow = 2L, ncol = 1L),
      delta_beta = matrix(1, nrow = 2L, ncol = 1L)
    ),
    "nrow$eta$"
  )

  expect_error(
    sgl_gradient_xt_residual_cpp(
      X = X,
      residual = matrix(1, nrow = 2L, ncol = 1L)
    ),
    "nrow$residual$"
  )

  X[1L, 1L] <- NA_real_

  expect_error(
    sgl_update_eta_cpp(
      X = X,
      eta = matrix(1, nrow = 3L, ncol = 1L),
      delta_beta = matrix(1, nrow = 2L, ncol = 1L)
    ),
    "finite"
  )
})
