reference_group_prox <- function(
  beta,
  group_index,
  group_weight,
  lambda_l1,
  lambda_group
) {
  result <- beta

  for (g in sort(unique(group_index[, 1L]))) {
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
test_that("group layout preserves original variable order within groups", {
  group_index <- matrix(
    c(2, 1, 2, 3, 1),
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  expect_true(is.matrix(layout$member_order))
  expect_true(is.matrix(layout$offsets))
  expect_true(is.matrix(layout$group_sizes))

  expect_identical(
    dim(layout$member_order),
    c(5L, 1L)
  )

  expect_identical(
    dim(layout$offsets),
    c(4L, 1L)
  )

  expect_identical(
    dim(layout$group_sizes),
    c(3L, 1L)
  )

  expect_equal(
    layout$member_order[, 1L],
    c(1, 4, 0, 2, 3),
    tolerance = 0
  )

  expect_equal(
    layout$offsets[, 1L],
    c(0, 2, 4, 5),
    tolerance = 0
  )

  expect_equal(
    layout$group_sizes[, 1L],
    c(2, 2, 1),
    tolerance = 0
  )
})


test_that("group layout metadata contain valid finite values", {
  group_index <- matrix(
    c(2, 1, 2, 3, 1),
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  expect_true(all(is.finite(layout$member_order)))
  expect_true(all(is.finite(layout$offsets)))
  expect_true(all(is.finite(layout$group_sizes)))

  expect_true(all(layout$member_order[, 1L] >= 0))
  expect_true(all(
    layout$member_order[, 1L] < nrow(group_index)
  ))

  expect_equal(
    layout$offsets[1L, 1L],
    0,
    tolerance = 0
  )

  expect_equal(
    layout$offsets[nrow(layout$offsets), 1L],
    nrow(group_index),
    tolerance = 0
  )
})


test_that("group layout has no uninitialized metadata values", {
  group_index <- matrix(
    c(2, 1, 2, 3, 1),
    ncol = 1L
  )

  layout <- sgl_group_layout_cpp(
    group_index
  )

  expect_false(anyNA(layout$member_order))
  expect_false(anyNA(layout$offsets))
  expect_false(anyNA(layout$group_sizes))

  expect_false(any(is.nan(layout$member_order)))
  expect_false(any(is.nan(layout$offsets)))
  expect_false(any(is.nan(layout$group_sizes)))

  expect_false(any(is.infinite(layout$member_order)))
  expect_false(any(is.infinite(layout$offsets)))
  expect_false(any(is.infinite(layout$group_sizes)))
})
