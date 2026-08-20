reference_group_prox_update_eta <- function(
  beta,
  eta,
  X,
  group_index,
  group_weight,
  lambda_l1,
  lambda_group
) {
  beta_new <- beta

  for (g in sort(unique(group_index[, 1L]))) {
    selected <- which(
      group_index[, 1L] == g
    )

    u <- sign(beta[selected, 1L]) *
      pmax(
        abs(beta[selected, 1L]) - lambda_l1,
        0
      )

    u_norm <- sqrt(sum(u * u))

    threshold <- lambda_group *
      group_weight[g, 1L]

    if (u_norm == 0 || u_norm <= threshold) {
      beta_new[selected, 1L] <- 0
    } else {
      beta_new[selected, 1L] <-
        u * (1 - threshold / u_norm)
    }
  }

  eta_new <- eta +
    X %*% (beta_new - beta)

  list(
    beta = beta_new,
    eta = eta_new
  )
}


test_that("combined group update preserves single-column matrix format", {
  set.seed(20240801)

  n <- 100L
  p <- 20L
  n_groups <- 5L

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

  eta <- matrix(
    rnorm(n),
    nrow = n,
    ncol = 1L
  )

  group_index <- matrix(
    sample(
      rep(seq_len(n_groups), length.out = p)
    ),
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  group_weight <- matrix(
    sqrt(tabulate(
      group_index[, 1L],
      nbins = n_groups
    )),
    ncol = 1L
  )

  actual <- sgl_sparse_group_prox_update_eta_cached_cpp(
    beta = beta,
    eta = eta,
    X = X,
    member_order = layout$member_order,
    offsets = layout$offsets,
    group_weight = group_weight,
    lambda_l1 = 0.15,
    lambda_group = 0.25
  )

  expect_true(is.list(actual))
  expect_true(is.matrix(actual$beta))
  expect_true(is.matrix(actual$eta))

  expect_identical(
    dim(actual$beta),
    c(p, 1L)
  )

  expect_identical(
    dim(actual$eta),
    c(n, 1L)
  )
})


test_that("combined group update matches R reference", {
  set.seed(20240802)

  for (n in c(1L, 5L, 50L, 200L)) {
    for (p in c(1L, 2L, 10L, 50L)) {
      n_groups <- min(5L, p)

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

      eta <- matrix(
        rnorm(n),
        nrow = n,
        ncol = 1L
      )

      groups <- sample(
        rep(seq_len(n_groups), length.out = p)
      )

      group_index <- matrix(
        groups,
        ncol = 1L
      )

      group_sizes <- tabulate(
        groups,
        nbins = n_groups
      )

      group_weight <- matrix(
        sqrt(group_sizes),
        ncol = 1L
      )

      layout <- sgl_group_layout_cpp(
        group_index
      )

      expected <- reference_group_prox_update_eta(
        beta = beta,
        eta = eta,
        X = X,
        group_index = group_index,
        group_weight = group_weight,
        lambda_l1 = 0.2,
        lambda_group = 0.3
      )

      actual <- sgl_sparse_group_prox_update_eta_cached_cpp(
        beta = beta,
        eta = eta,
        X = X,
        member_order = layout$member_order,
        offsets = layout$offsets,
        group_weight = group_weight,
        lambda_l1 = 0.2,
        lambda_group = 0.3
      )

      expect_true(is.matrix(actual$beta))
      expect_true(is.matrix(actual$eta))

      expect_lt(
        max(abs(actual$beta - expected$beta)),
        1e-12
      )

      expect_lt(
        max(abs(actual$eta - expected$eta)),
        1e-8
      )
    }
  }
})


test_that("zero penalties leave beta and eta unchanged", {
  set.seed(20240804)

  n <- 100L
  p <- 30L
  n_groups <- 6L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- matrix(
    rnorm(p),
    ncol = 1L
  )

  eta <- matrix(
    rnorm(n),
    ncol = 1L
  )

  groups <- sample(
    rep(seq_len(n_groups), length.out = p)
  )

  group_index <- matrix(
    groups,
    ncol = 1L
  )

  group_weight <- matrix(
    sqrt(tabulate(
      groups,
      nbins = n_groups
    )),
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  actual <- sgl_sparse_group_prox_update_eta_cached_cpp(
    beta = beta,
    eta = eta,
    X = X,
    member_order = layout$member_order,
    offsets = layout$offsets,
    group_weight = group_weight,
    lambda_l1 = 0,
    lambda_group = 0
  )

  expect_identical(actual$beta, beta)
  expect_identical(actual$eta, eta)
})


test_that("large penalties produce zero beta and eta updates correctly", {
  set.seed(20240805)

  n <- 50L
  p <- 10L
  n_groups <- 2L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- matrix(
    rnorm(p),
    ncol = 1L
  )

  eta <- matrix(
    rnorm(n),
    ncol = 1L
  )

  groups <- rep(
    seq_len(n_groups),
    length.out = p
  )

  group_index <- matrix(
    groups,
    ncol = 1L
  )

  group_weight <- matrix(
    1,
    nrow = n_groups,
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  actual <- sgl_sparse_group_prox_update_eta_cached_cpp(
    beta = beta,
    eta = eta,
    X = X,
    member_order = layout$member_order,
    offsets = layout$offsets,
    group_weight = group_weight,
    lambda_l1 = 100,
    lambda_group = 100
  )

  expected_beta <- matrix(
    0,
    nrow = p,
    ncol = 1L
  )

  expected_eta <- eta -
    X %*% beta

  expect_lt(
    max(abs(actual$beta - expected_beta)),
    1e-12
  )

  expect_lt(
    max(abs(actual$eta - expected_eta)),
    1e-8
  )
})


test_that("non-contiguous groups are handled correctly", {
  set.seed(20240806)

  n <- 80L
  p <- 12L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  beta <- matrix(
    rnorm(p),
    ncol = 1L
  )

  eta <- matrix(
    rnorm(n),
    ncol = 1L
  )

  group_index <- matrix(
    c(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
    ncol = 1L
  )

  group_weight <- matrix(
    sqrt(6),
    nrow = 2L,
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  expected <- reference_group_prox_update_eta(
    beta = beta,
    eta = eta,
    X = X,
    group_index = group_index,
    group_weight = group_weight,
    lambda_l1 = 0.25,
    lambda_group = 0.35
  )

  actual <- sgl_sparse_group_prox_update_eta_cached_cpp(
    beta = beta,
    eta = eta,
    X = X,
    member_order = layout$member_order,
    offsets = layout$offsets,
    group_weight = group_weight,
    lambda_l1 = 0.25,
    lambda_group = 0.35
  )

  expect_lt(
    max(abs(actual$beta - expected$beta)),
    1e-12
  )

  expect_lt(
    max(abs(actual$eta - expected$eta)),
    1e-8
  )
})


test_that("invalid dimensions and values are rejected", {
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

  eta <- matrix(
    1,
    nrow = 5L,
    ncol = 1L
  )

  member_order <- matrix(
    c(0, 1, 2),
    ncol = 1L
  )

  offsets <- matrix(
    c(0, 1, 3),
    ncol = 1L
  )

  group_weight <- matrix(
    c(1, 1),
    ncol = 1L
  )

  expect_error(
    sgl_sparse_group_prox_update_eta_cached_cpp(
      beta = matrix(1, nrow = 2L, ncol = 1L),
      eta = eta,
      X = X,
      member_order = member_order,
      offsets = offsets,
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "nrow"
  )

  expect_error(
    sgl_sparse_group_prox_update_eta_cached_cpp(
      beta = beta,
      eta = matrix(1, nrow = 4L, ncol = 1L),
      X = X,
      member_order = member_order,
      offsets = offsets,
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "nrow"
  )

  expect_error(
    sgl_sparse_group_prox_update_eta_cached_cpp(
      beta = beta,
      eta = eta,
      X = X,
      member_order = matrix(c(0, 1), ncol = 1L),
      offsets = offsets,
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "nrow"
  )

  expect_error(
    sgl_sparse_group_prox_update_eta_cached_cpp(
      beta = beta,
      eta = eta,
      X = X,
      member_order = member_order,
      offsets = matrix(c(0, 3, 3), ncol = 1L),
      group_weight = group_weight,
      lambda_l1 = 0.1,
      lambda_group = 0.2
    ),
    "at least one"
  )

  expect_error(
    sgl_sparse_group_prox_update_eta_cached_cpp(
      beta = beta,
      eta = eta,
      X = X,
      member_order = member_order,
      offsets = offsets,
      group_weight = group_weight,
      lambda_l1 = -1,
      lambda_group = 0.2
    ),
    "non-negative"
  )
})
