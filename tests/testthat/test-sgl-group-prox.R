reference_group_prox <- function(
  beta,
  group_index,
  group_weight,
  lambda_l1,
  lambda_group
) {
  stopifnot(
    is.matrix(beta),
    ncol(beta) == 1L,
    is.matrix(group_index),
    ncol(group_index) == 1L,
    is.matrix(group_weight),
    ncol(group_weight) == 1L
  )

  result <- beta

  groups <- sort(unique(group_index[, 1L]))

  for (g in groups) {
    selected <- which(group_index[, 1L] == g)

    u <- sign(beta[selected, 1L]) *
      pmax(abs(beta[selected, 1L]) - lambda_l1, 0)

    u_norm <- sqrt(sum(u * u))
    threshold <- lambda_group *
      group_weight[g, 1L]

    if (u_norm == 0 || u_norm <= threshold) {
      result[selected, 1L] <- 0
    } else {
      result[selected, 1L] <-
        u * (1 - threshold / u_norm)
    }
  }

  result
}


make_group_index <- function(groups) {
  matrix(
    as.numeric(groups),
    ncol = 1L
  )
}


test_that("group proximal update preserves matrix shape", {
  beta <- matrix(
    c(1, -2, 3, -4, 5),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2, 3)
  )

  group_weight <- matrix(
    c(1, sqrt(2), 1),
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.1,
    lambda_group = 0.2
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), dim(beta))
})


test_that("group proximal update matches R reference", {
  set.seed(20240611)

  for (p in c(1L, 2L, 5L, 20L, 100L)) {
    for (n_groups in c(1L, 2L, 5L)) {
      if (n_groups > p) {
        next
      }

      groups <- sample(
        rep(seq_len(n_groups), length.out = p)
      )

      beta <- matrix(
        rnorm(p),
        nrow = p,
        ncol = 1L
      )

      group_index <- make_group_index(groups)

      group_sizes <- tabulate(
        groups,
        nbins = n_groups
      )

      group_weight <- matrix(
        sqrt(group_sizes),
        nrow = n_groups,
        ncol = 1L
      )

      lambda_l1 <- runif(1L, 0, 1)
      lambda_group <- runif(1L, 0, 1)

      expected <- reference_group_prox(
        beta = beta,
        group_index = group_index,
        group_weight = group_weight,
        lambda_l1 = lambda_l1,
        lambda_group = lambda_group
      )

      actual <- sgl_sparse_group_prox_groups_cpp(
        beta = beta,
        group_index = group_index,
        group_weight = group_weight,
        lambda_l1 = lambda_l1,
        lambda_group = lambda_group
      )

      expect_true(is.matrix(actual))
      expect_identical(dim(actual), c(p, 1L))
      expect_lt(
        max(abs(actual - expected)),
        1e-12
      )
    }
  }
})


test_that("group proximal update supports non-contiguous group members", {
  beta <- matrix(
    c(3, -2, 4, -5, 6, -7),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 2, 1, 2, 1, 2)
  )

  group_weight <- matrix(
    c(sqrt(3), sqrt(3)),
    ncol = 1L
  )

  expected <- reference_group_prox(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.25,
    lambda_group = 0.3
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.25,
    lambda_group = 0.3
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-12
  )
})


test_that("zero L1 and group penalties preserve beta", {
  set.seed(20240612)

  beta <- matrix(
    rnorm(50L),
    ncol = 1L
  )

  group_index <- make_group_index(
    rep(1:5, each = 10L)
  )

  group_weight <- matrix(
    sqrt(10),
    nrow = 5L,
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0,
    lambda_group = 0
  )

  expect_true(is.matrix(actual))
  expect_identical(dim(actual), dim(beta))
  expect_lt(
    max(abs(actual - beta)),
    1e-12
  )
})


test_that("large L1 penalty produces an all-zero result", {
  beta <- matrix(
    c(1, -2, 3, -4, 5),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2, 3)
  )

  group_weight <- matrix(
    c(1, sqrt(2), 1),
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 100,
    lambda_group = 0
  )

  expect_identical(
    actual,
    matrix(0, nrow = 5L, ncol = 1L)
  )
})


test_that("large group penalty produces an all-zero result", {
  beta <- matrix(
    c(1, -2, 3, -4, 5),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2, 3)
  )

  group_weight <- matrix(
    c(1, sqrt(2), 1),
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0,
    lambda_group = 100
  )

  expect_identical(
    actual,
    matrix(0, nrow = 5L, ncol = 1L)
  )
})


test_that("group boundary values produce zero groups", {
  beta <- matrix(
    c(3, 4, 1, 0),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2)
  )

  group_weight <- matrix(
    c(1, 1),
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0,
    lambda_group = 5
  )

  expect_identical(
    actual,
    matrix(0, nrow = 4L, ncol = 1L)
  )
})


test_that("single-variable groups are handled correctly", {
  set.seed(20240613)

  beta <- matrix(
    rnorm(10L),
    ncol = 1L
  )

  group_index <- make_group_index(
    seq_len(10L)
  )

  group_weight <- matrix(
    1,
    nrow = 10L,
    ncol = 1L
  )

  expected <- reference_group_prox(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.1,
    lambda_group = 0.2
  )

  actual <- sgl_sparse_group_prox_groups_cpp(
    beta = beta,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.1,
    lambda_group = 0.2
  )

  expect_lt(
    max(abs(actual - expected)),
    1e-12
  )
})


test_that("group index must be a single-column matrix", {
  beta <- matrix(
    rnorm(4L),
    ncol = 1L
  )

  group_index <- matrix(
    c(1, 1, 2, 2),
    nrow = 2L,
    ncol = 2L
  )

  group_weight <- matrix(
    c(1, 1),
    ncol = 1L
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = group_index,
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "single-column matrix"
  )
})


test_that("group index must contain consecutive positive integers", {
  beta <- matrix(
    rnorm(4L),
    ncol = 1L
  )

  group_weight <- matrix(
    c(1, 1),
    ncol = 1L
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = make_group_index(c(1, 1, 3, 3)),
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "consecutive"
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = make_group_index(c(0, 1, 1, 2)),
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "positive integers"
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = make_group_index(c(1, 1.5, 2, 2)),
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "positive integers"
  )
})


test_that("group dimensions and penalties are validated", {
  beta <- matrix(
    rnorm(4L),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2)
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = group_index,
      group_weight = matrix(1, nrow = 1L, ncol = 1L),
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "number of groups"
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = group_index,
      group_weight = matrix(c(1, 1), ncol = 1L),
      lambda_l1 = -1,
      lambda_group = 0.2
    ),
    "finite and non-negative"
  )

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta = beta,
      group_index = group_index,
      group_weight = matrix(c(1, 1), ncol = 1L),
      lambda_l1 = 0.1,
      lambda_group = -1
    ),
    "finite and non-negative"
  )
})


test_that("non-finite inputs are rejected", {
  beta <- matrix(
    rnorm(4L),
    ncol = 1L
  )

  group_index <- make_group_index(
    c(1, 1, 2, 2)
  )

  group_weight <- matrix(
    c(1, 1),
    ncol = 1L
  )

  beta[1L, 1L] <- NA_real_

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta,
      group_index,
      group_weight,
      0.1,
      0.2
    ),
    "finite"
  )

  beta[1L, 1L] <- 1
  group_index[2L, 1L] <- Inf

  expect_error(
    sgl_sparse_group_prox_groups_cpp(
      beta,
      group_index,
      group_weight,
      0.1,
      0.2
    ),
    "finite values"
  )
})
