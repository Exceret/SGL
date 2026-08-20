test_that("linear predictor initialization returns a single-column matrix", {
  set.seed(20240901)

  n <- 100L
  p <- 20L

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

  intercept <- matrix(
    0.35,
    nrow = 1L,
    ncol = 1L
  )

  actual <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept
  )

  expected <- intercept[1L, 1L] +
    X %*% beta

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(n, 1L))
  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("linear predictor initialization matches R reference", {
  set.seed(20240902)

  for (n in c(1L, 2L, 10L, 100L, 1000L)) {
    for (p in c(1L, 2L, 5L, 20L, 100L)) {
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

      intercept <- matrix(
        rnorm(1L),
        nrow = 1L,
        ncol = 1L
      )

      expected <- intercept[1L, 1L] +
        X %*% beta

      actual <- sgl_initialize_linear_predictor_cpp(
        X = X,
        beta = beta,
        intercept = intercept
      )

      expect_true(is.matrix(actual))
      expect_identical(dim(actual), c(n, 1L))
      expect_true(all(is.finite(actual)))

      expect_lt(
        max(abs(actual - expected)),
        1e-8
      )
    }
  }
})


test_that("zero beta produces a constant linear predictor", {
  set.seed(20240903)

  n <- 100L
  p <- 20L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  intercept <- matrix(
    -1.25,
    nrow = 1L,
    ncol = 1L
  )

  actual <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept
  )

  expected <- matrix(
    intercept[1L, 1L],
    nrow = n,
    ncol = 1L
  )

  expect_identical(actual, expected)
})


test_that("zero intercept produces X beta", {
  set.seed(20240904)

  n <- 100L
  p <- 20L

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

  intercept <- matrix(
    0,
    nrow = 1L,
    ncol = 1L
  )

  actual <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept
  )

  expected <- X %*% beta

  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("linear predictor state update matches full recomputation", {
  set.seed(20240905)

  for (n in c(1L, 10L, 100L, 1000L)) {
    for (p in c(1L, 2L, 10L, 50L)) {
      X <- matrix(
        rnorm(n * p),
        nrow = n,
        ncol = p
      )

      beta_old <- matrix(
        rnorm(p),
        nrow = p,
        ncol = 1L
      )

      beta_new <- matrix(
        rnorm(p),
        nrow = p,
        ncol = 1L
      )

      intercept_old <- matrix(
        rnorm(1L),
        nrow = 1L,
        ncol = 1L
      )

      intercept_new <- matrix(
        rnorm(1L),
        nrow = 1L,
        ncol = 1L
      )

      eta_old <- sgl_initialize_linear_predictor_cpp(
        X = X,
        beta = beta_old,
        intercept = intercept_old
      )

      expected <- sgl_initialize_linear_predictor_cpp(
        X = X,
        beta = beta_new,
        intercept = intercept_new
      )

      actual <- sgl_update_linear_predictor_state_cpp(
        eta = eta_old,
        X = X,
        beta_old = beta_old,
        beta_new = beta_new,
        intercept_old = intercept_old,
        intercept_new = intercept_new
      )

      expect_true(is.matrix(actual))
      expect_identical(dim(actual), c(n, 1L))
      expect_lt(
        max(abs(actual - expected)),
        1e-8
      )
    }
  }
})


test_that("state update handles beta and intercept unchanged", {
  set.seed(20240906)

  n <- 100L
  p <- 20L

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

  intercept <- matrix(
    0.5,
    nrow = 1L,
    ncol = 1L
  )

  eta <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept
  )

  actual <- sgl_update_linear_predictor_state_cpp(
    eta = eta,
    X = X,
    beta_old = beta,
    beta_new = beta,
    intercept_old = intercept,
    intercept_new = intercept
  )

  expect_identical(actual, eta)
})


test_that("state update handles intercept-only changes", {
  set.seed(20240907)

  n <- 100L
  p <- 20L

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

  intercept_old <- matrix(
    0.25,
    nrow = 1L,
    ncol = 1L
  )

  intercept_new <- matrix(
    -0.75,
    nrow = 1L,
    ncol = 1L
  )

  eta <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept_old
  )

  actual <- sgl_update_linear_predictor_state_cpp(
    eta = eta,
    X = X,
    beta_old = beta,
    beta_new = beta,
    intercept_old = intercept_old,
    intercept_new = intercept_new
  )

  expected <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept_new
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-12
  )
})


test_that("state update handles beta-only changes", {
  set.seed(20240908)

  n <- 100L
  p <- 20L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta_old <- matrix(
    rnorm(p),
    nrow = p,
    ncol = 1L
  )

  beta_new <- matrix(
    rnorm(p),
    nrow = p,
    ncol = 1L
  )

  intercept <- matrix(
    0.5,
    nrow = 1L,
    ncol = 1L
  )

  eta <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta_old,
    intercept = intercept
  )

  actual <- sgl_update_linear_predictor_state_cpp(
    eta = eta,
    X = X,
    beta_old = beta_old,
    beta_new = beta_new,
    intercept_old = intercept,
    intercept_new = intercept
  )

  expected <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta_new,
    intercept = intercept
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-8
  )
})


test_that("linear predictor initialization handles single observation", {
  X <- matrix(
    c(2, -3, 4),
    nrow = 1L,
    ncol = 3L
  )

  beta <- matrix(
    c(0.5, -1, 2),
    nrow = 3L,
    ncol = 1L
  )

  intercept <- matrix(
    1.5,
    nrow = 1L,
    ncol = 1L
  )

  actual <- sgl_initialize_linear_predictor_cpp(
    X = X,
    beta = beta,
    intercept = intercept
  )

  expected <- matrix(
    1.5 + sum(X[1L, ] * beta[, 1L]),
    nrow = 1L,
    ncol = 1L
  )

  expect_identical(dim(actual), c(1L, 1L))
  expect_lt(
    max(abs(actual - expected)),
    1e-12
  )
})


test_that("invalid linear predictor dimensions are rejected", {
  X <- matrix(
    1,
    nrow = 5L,
    ncol = 3L
  )

  expect_error(
    sgl_initialize_linear_predictor_cpp(
      X = X,
      beta = matrix(1, nrow = 2L, ncol = 1L),
      intercept = matrix(0, nrow = 1L, ncol = 1L)
    ),
    "nrow"
  )

  expect_error(
    sgl_initialize_linear_predictor_cpp(
      X = X,
      beta = matrix(1, nrow = 3L, ncol = 1L),
      intercept = matrix(0, nrow = 2L, ncol = 1L)
    ),
    "1 x 1"
  )

  expect_error(
    sgl_update_linear_predictor_state_cpp(
      eta = matrix(1, nrow = 4L, ncol = 1L),
      X = X,
      beta_old = matrix(1, nrow = 3L, ncol = 1L),
      beta_new = matrix(1, nrow = 3L, ncol = 1L),
      intercept_old = matrix(0, nrow = 1L, ncol = 1L),
      intercept_new = matrix(0, nrow = 1L, ncol = 1L)
    ),
    "nrow"
  )

  expect_error(
    sgl_update_linear_predictor_state_cpp(
      eta = matrix(1, nrow = 5L, ncol = 1L),
      X = X,
      beta_old = matrix(1, nrow = 2L, ncol = 1L),
      beta_new = matrix(1, nrow = 3L, ncol = 1L),
      intercept_old = matrix(0, nrow = 1L, ncol = 1L),
      intercept_new = matrix(0, nrow = 1L, ncol = 1L)
    ),
    "beta_old"
  )
})


test_that("non-finite linear predictor inputs are rejected", {
  X <- matrix(
    1,
    nrow = 5L,
    ncol = 3L
  )

  beta <- matrix(
    1,
    nrow = 3L,
    ncol = 1L
  )

  intercept <- matrix(
    0,
    nrow = 1L,
    ncol = 1L
  )

  X[1L, 1L] <- NA_real_

  expect_error(
    sgl_initialize_linear_predictor_cpp(
      X = X,
      beta = beta,
      intercept = intercept
    ),
    "finite"
  )

  X[1L, 1L] <- 1
  beta[1L, 1L] <- Inf

  expect_error(
    sgl_initialize_linear_predictor_cpp(
      X = X,
      beta = beta,
      intercept = intercept
    ),
    "finite"
  )

  beta[1L, 1L] <- 1
  intercept[1L, 1L] <- NaN

  expect_error(
    sgl_initialize_linear_predictor_cpp(
      X = X,
      beta = beta,
      intercept = intercept
    ),
    "finite"
  )
})
