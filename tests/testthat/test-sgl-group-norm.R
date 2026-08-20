test_that("group norm returns a scalar", {
  z <- matrix(
    c(3, 4),
    nrow = 2L,
    ncol = 1L
  )

  actual <- sgl_group_l2_norm_cpp(z)

  expect_type(actual, "double")
  expect_length(actual, 1L)
  expect_equal(actual, 5)
})


test_that("group norm matches R reference", {
  set.seed(20240601)

  for (m in c(0L, 1L, 2L, 5L, 20L, 100L, 1000L)) {
    for (iter in seq_len(100L)) {
      z <- matrix(
        rnorm(m),
        nrow = m,
        ncol = 1L
      )

      expected <- sqrt(sum(z * z))
      actual <- sgl_group_l2_norm_cpp(z)

      expect_true(is.finite(actual))
      expect_lt(
        abs(actual - expected),
        1e-12
      )
    }
  }
})


test_that("zero vector has zero group norm", {
  z <- matrix(
    0,
    nrow = 100L,
    ncol = 1L
  )

  actual <- sgl_group_l2_norm_cpp(z)

  expect_identical(actual, 0)
})


test_that("group norm is invariant to sign", {
  set.seed(20240602)

  z <- matrix(
    rnorm(100L),
    nrow = 100L,
    ncol = 1L
  )

  actual_1 <- sgl_group_l2_norm_cpp(z)
  actual_2 <- sgl_group_l2_norm_cpp(-z)

  expect_lt(
    abs(actual_1 - actual_2),
    1e-12
  )
})


test_that("group norm is homogeneous for positive scalars", {
  set.seed(20240603)

  z <- matrix(
    rnorm(100L),
    nrow = 100L,
    ncol = 1L
  )

  scale_value <- 3.75

  actual <- sgl_group_l2_norm_cpp(
    scale_value * z
  )

  expected <- scale_value *
    sgl_group_l2_norm_cpp(z)

  expect_lt(
    abs(actual - expected),
    1e-10
  )
})


test_that("group norm is zero for an empty single-column matrix", {
  z <- matrix(
    numeric(0),
    nrow = 0L,
    ncol = 1L
  )

  actual <- sgl_group_l2_norm_cpp(z)

  expect_type(actual, "double")
  expect_length(actual, 1L)
  expect_identical(actual, 0)
})


test_that("group norm handles a one-element group", {
  values <- c(-10, -1, 0, 1, 10)

  for (value in values) {
    z <- matrix(
      value,
      nrow = 1L,
      ncol = 1L
    )

    actual <- sgl_group_l2_norm_cpp(z)

    expect_equal(
      actual,
      abs(value),
      tolerance = 1e-12
    )
  }
})


test_that("group norm rejects non-column matrices", {
  z <- matrix(
    rnorm(6L),
    nrow = 2L,
    ncol = 3L
  )

  expect_error(
    sgl_group_l2_norm_cpp(z),
    "single-column matrix"
  )
})


test_that("group norm rejects non-finite values", {
  z <- matrix(
    c(1, NA_real_, 3),
    ncol = 1L
  )

  expect_error(
    sgl_group_l2_norm_cpp(z),
    "finite"
  )

  z[2L, 1L] <- NaN

  expect_error(
    sgl_group_l2_norm_cpp(z),
    "finite"
  )

  z[2L, 1L] <- Inf

  expect_error(
    sgl_group_l2_norm_cpp(z),
    "finite"
  )

  z[2L, 1L] <- -Inf

  expect_error(
    sgl_group_l2_norm_cpp(z),
    "finite"
  )
})
