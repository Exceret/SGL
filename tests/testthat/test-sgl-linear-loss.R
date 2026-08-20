reference_linear_squared_loss <- function(eta, y) {
  stopifnot(
    is.matrix(eta),
    ncol(eta) == 1L,
    is.matrix(y),
    ncol(y) == 1L,
    nrow(eta) == nrow(y)
  )

  0.5 * sum((eta - y)^2)
}


test_that("linear squared loss returns a scalar", {
  eta <- matrix(
    c(-2, -1, 0, 1, 2),
    ncol = 1L
  )

  y <- matrix(
    c(0, 0.5, 1, 1.5, 2),
    ncol = 1L
  )

  actual <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  expect_type(actual, "double")
  expect_length(actual, 1L)
  expect_true(is.finite(actual))
})


test_that("linear squared loss matches R reference", {
  set.seed(20240501)

  for (n in c(1L, 2L, 10L, 100L, 1000L, 10000L)) {
    for (iter in seq_len(50L)) {
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

      expected <- reference_linear_squared_loss(
        eta = eta,
        y = y
      )

      actual <- sgl_linear_squared_loss_cpp(
        eta = eta,
        y = y
      )

      expect_true(is.finite(actual))
      expect_lt(
        abs(actual - expected),
        1e-8
      )
    }
  }
})


test_that("zero residual produces zero loss", {
  set.seed(20240502)

  eta <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  actual <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = eta
  )

  expect_identical(actual, 0)
})


test_that("loss is non-negative", {
  set.seed(20240503)

  eta <- matrix(
    rnorm(1000L, sd = 20),
    ncol = 1L
  )

  y <- matrix(
    rnorm(1000L, sd = 20),
    ncol = 1L
  )

  actual <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  expect_true(is.finite(actual))
  expect_gte(actual, 0)
})


test_that("loss is symmetric in the residual direction", {
  set.seed(20240504)

  eta <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  y <- matrix(
    rnorm(100L),
    ncol = 1L
  )

  actual_1 <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  actual_2 <- sgl_linear_squared_loss_cpp(
    eta = y,
    y = eta
  )

  expect_lt(
    abs(actual_1 - actual_2),
    1e-12
  )
})


test_that("loss is additive over observations", {
  set.seed(20240505)

  n <- 200L

  eta <- matrix(
    rnorm(n, sd = 5),
    ncol = 1L
  )

  y <- matrix(
    rnorm(n, sd = 5),
    ncol = 1L
  )

  full <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  first <- sgl_linear_squared_loss_cpp(
    eta = eta[1:100, , drop = FALSE],
    y = y[1:100, , drop = FALSE]
  )

  second <- sgl_linear_squared_loss_cpp(
    eta = eta[101:200, , drop = FALSE],
    y = y[101:200, , drop = FALSE]
  )

  expect_lt(
    abs(full - (first + second)),
    1e-8
  )
})


test_that("loss is consistent with finite differences", {
  set.seed(20240506)

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

  objective <- function(beta_value) {
    eta_value <- matrix(
      intercept + X %*% beta_value,
      nrow = n,
      ncol = 1L
    )

    sgl_linear_squared_loss_cpp(
      eta = eta_value,
      y = y
    )
  }

  analytic <- sgl_linear_gradient_cpp(
    X = X,
    eta = matrix(
      intercept + X %*% beta,
      nrow = n,
      ncol = 1L
    ),
    y = y
  )

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


test_that("loss handles empty single-column matrices", {
  eta <- matrix(
    numeric(0),
    nrow = 0L,
    ncol = 1L
  )

  y <- matrix(
    numeric(0),
    nrow = 0L,
    ncol = 1L
  )

  actual <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  expect_type(actual, "double")
  expect_length(actual, 1L)
  expect_identical(actual, 0)
})


test_that("loss handles large finite values", {
  eta <- matrix(
    c(
      -1e100,
      -1e50,
      0,
      1e50,
      1e100
    ),
    ncol = 1L
  )

  y <- matrix(
    c(
      1e100,
      1e50,
      0,
      -1e50,
      -1e100
    ),
    ncol = 1L
  )

  actual <- sgl_linear_squared_loss_cpp(
    eta = eta,
    y = y
  )

  expect_true(is.finite(actual))
  expect_gte(actual, 0)
})


test_that("non-column matrix inputs are rejected", {
  eta <- matrix(
    rnorm(6L),
    nrow = 2L,
    ncol = 3L
  )

  y <- matrix(
    c(0, 1),
    ncol = 1L
  )

  expect_error(
    sgl_linear_squared_loss_cpp(
      eta = eta,
      y = y
    ),
    "single-column matrix"
  )

  eta <- matrix(
    c(0, 1),
    ncol = 1L
  )

  y <- matrix(
    rnorm(6L),
    nrow = 2L,
    ncol = 3L
  )

  expect_error(
    sgl_linear_squared_loss_cpp(
      eta = eta,
      y = y
    ),
    "single-column matrix"
  )
})


test_that("non-finite values are rejected", {
  eta <- matrix(
    c(0, NA_real_, 1),
    ncol = 1L
  )

  y <- matrix(
    c(0, 1, 0),
    ncol = 1L
  )

  expect_error(
    sgl_linear_squared_loss_cpp(
      eta = eta,
      y = y
    ),
    "finite"
  )

  eta <- matrix(
    c(0, 1, Inf),
    ncol = 1L
  )

  expect_error(
    sgl_linear_squared_loss_cpp(
      eta = eta,
      y = y
    ),
    "finite"
  )

  eta <- matrix(
    c(0, 1, 0),
    ncol = 1L
  )

  y[2L, 1L] <- NaN

  expect_error(
    sgl_linear_squared_loss_cpp(
      eta = eta,
      y = y
    ),
    "finite"
  )
})
