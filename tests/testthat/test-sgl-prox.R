reference_sparse_group_prox <- function(
  z,
  lambda_l1,
  lambda_group,
  group_weight = 1
) {
  stopifnot(is.matrix(z), ncol(z) == 1L)

  u <- sign(z) * pmax(abs(z) - lambda_l1, 0)

  u_norm <- sqrt(sum(u * u))
  group_threshold <- lambda_group * group_weight

  if (u_norm == 0 || u_norm <= group_threshold) {
    return(matrix(
      0,
      nrow = nrow(z),
      ncol = 1L
    ))
  }

  u * (1 - group_threshold / u_norm)
}


test_that("sparse-group proximal operator preserves matrix shape", {
  set.seed(20240311)

  for (m in c(1L, 2L, 5L, 20L, 100L)) {
    z <- matrix(rnorm(m), nrow = m, ncol = 1L)

    actual <- sgl_sparse_group_prox_cpp(
      z = z,
      lambda_l1 = 0.2,
      lambda_group = 0.3,
      group_weight = sqrt(m)
    )

    expect_true(is.matrix(actual))
    expect_identical(dim(actual), c(m, 1L))
  }
})


test_that("sparse-group proximal operator matches R reference", {
  set.seed(20240312)

  for (m in c(1L, 2L, 5L, 20L, 100L)) {
    for (iter in seq_len(100L)) {
      z <- matrix(rnorm(m), nrow = m, ncol = 1L)

      lambda_l1 <- runif(1L, 0, 1)
      lambda_group <- runif(1L, 0, 1)
      group_weight <- runif(1L, 0, 3)

      expected <- reference_sparse_group_prox(
        z = z,
        lambda_l1 = lambda_l1,
        lambda_group = lambda_group,
        group_weight = group_weight
      )

      actual <- sgl_sparse_group_prox_cpp(
        z = z,
        lambda_l1 = lambda_l1,
        lambda_group = lambda_group,
        group_weight = group_weight
      )

      expect_true(is.matrix(actual))
      expect_identical(dim(actual), c(m, 1L))
      expect_lt(
        max(abs(actual - expected)),
        1e-12
      )
    }
  }
})


test_that("zero vector remains zero", {
  z <- matrix(0, nrow = 10L, ncol = 1L)

  actual <- sgl_sparse_group_prox_cpp(
    z = z,
    lambda_l1 = 0.1,
    lambda_group = 0.2,
    group_weight = sqrt(10)
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), c(10L, 1L))
  expect_identical(actual, z)
})


test_that("large L1 threshold produces zero vector", {
  z <- matrix(c(1, -2, 3, -4), ncol = 1L)

  actual <- sgl_sparse_group_prox_cpp(
    z = z,
    lambda_l1 = 10,
    lambda_group = 0,
    group_weight = 1
  )

  expect_identical(
    actual,
    matrix(0, nrow = 4L, ncol = 1L)
  )
})


test_that("large group threshold produces zero vector", {
  z <- matrix(c(1, -2, 3, -4), ncol = 1L)

  actual <- sgl_sparse_group_prox_cpp(
    z = z,
    lambda_l1 = 0,
    lambda_group = 100,
    group_weight = 1
  )

  expect_identical(
    actual,
    matrix(0, nrow = 4L, ncol = 1L)
  )
})


test_that("boundary at group threshold produces zero vector", {
  z <- matrix(c(3, 4), ncol = 1L)

  actual <- sgl_sparse_group_prox_cpp(
    z = z,
    lambda_l1 = 0,
    lambda_group = 5,
    group_weight = 1
  )

  expect_identical(
    actual,
    matrix(0, nrow = 2L, ncol = 1L)
  )
})


test_that("zero penalties return the original matrix", {
  set.seed(20240313)

  z <- matrix(rnorm(30L), ncol = 1L)

  actual <- sgl_sparse_group_prox_cpp(
    z = z,
    lambda_l1 = 0,
    lambda_group = 0,
    group_weight = 1
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), dim(z))
  expect_lt(max(abs(actual - z)), 1e-12)
})


test_that("negative and non-finite penalties are rejected", {
  z <- matrix(rnorm(5L), ncol = 1L)

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = -1,
      lambda_group = 0,
      group_weight = 1
    ),
    "finite non-negative"
  )

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = 0,
      lambda_group = -1,
      group_weight = 1
    ),
    "finite non-negative"
  )

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = 0,
      lambda_group = 0,
      group_weight = -1
    ),
    "finite non-negative"
  )

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = Inf,
      lambda_group = 0,
      group_weight = 1
    ),
    "finite non-negative"
  )

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = 0,
      lambda_group = NaN,
      group_weight = 1
    ),
    "finite non-negative"
  )
})


test_that("non-column matrix input is rejected", {
  z <- matrix(rnorm(6L), nrow = 2L, ncol = 3L)

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = 0.1,
      lambda_group = 0.2,
      group_weight = 1
    ),
    "single-column matrix"
  )
})


test_that("non-finite coefficient input is rejected", {
  z <- matrix(rnorm(5L), ncol = 1L)
  z[2L, 1L] <- NA_real_

  expect_error(
    sgl_sparse_group_prox_cpp(
      z,
      lambda_l1 = 0.1,
      lambda_group = 0.2,
      group_weight = 1
    ),
    "finite"
  )
})
