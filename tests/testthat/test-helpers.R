test_that("C++ lambda max matches R calculation", {
  set.seed(20250501)

  gradient <- matrix(
    c(
      2,
      -1,
      0.5,
      -3,
      1
    ),
    ncol = 1L
  )

  group_index <- matrix(
    c(1, 1, 2, 2, 3),
    ncol = 1L
  )

  group_weight <- matrix(
    c(
      sqrt(2),
      sqrt(2),
      1
    ),
    ncol = 1L
  )

  alpha <- 0.5

  actual <- sgl_lambda_max_from_gradient_cpp(
    gradient = gradient,
    group_index = group_index,
    group_weight = group_weight,
    alpha = alpha
  )

  expected <- 0

  expected <- max(
    max(abs(gradient)) / alpha,
    expected
  )

  for (g in 1:3) {
    selected <- which(
      group_index[, 1L] == g
    )

    expected <- max(
      expected,
      sqrt(sum(
        gradient[selected, 1L]^2
      )) /
        ((1 - alpha) *
          group_weight[g, 1L])
    )
  }

  expect_equal(
    actual,
    expected,
    tolerance = 1e-12
  )
})


test_that("C++ lambda path respects supplied values", {
  lambdas <- c(
    0.2,
    0.1,
    0.05
  )

  actual <- sgl_make_lambda_path_cpp(
    lambda_input = lambdas,
    lambda_max = 1,
    n_lambda = 20L,
    min_frac = 0.1,
    lambda_decay = 0.8
  )

  expect_identical(
    actual,
    lambdas
  )
})


test_that("C++ lambda path matches expected geometric path", {
  actual <- sgl_make_lambda_path_cpp(
    lambda_input = NULL,
    lambda_max = 1,
    n_lambda = 3L,
    min_frac = 0.01,
    lambda_decay = 0.5
  )

  expected <- c(
    1,
    0.5,
    0.25
  )

  expect_equal(
    actual,
    expected,
    tolerance = 1e-12
  )
})


test_that("C++ Cox response parser handles two-column matrix", {
  response <- cbind(
    c(5, 4, 3, 2),
    c(1, 0, 1, 0)
  )

  actual <- sgl_parse_cox_response_cpp(
    response
  )

  expect_identical(
    dim(actual$time),
    c(4L, 1L)
  )

  expect_identical(
    dim(actual$status),
    c(4L, 1L)
  )

  expect_equal(
    actual$time[, 1L],
    response[, 1L]
  )

  expect_equal(
    actual$status[, 1L],
    response[, 2L]
  )
})


test_that("C++ Cox response parser handles list", {
  response <- list(
    time = c(5, 4, 3, 2),
    status = c(1, 0, 1, 0)
  )

  actual <- sgl_parse_cox_response_cpp(
    response
  )

  expect_equal(
    actual$time[, 1L],
    response$time
  )

  expect_equal(
    actual$status[, 1L],
    response$status
  )
})


test_that("C++ Cox response parser rejects no-event data", {
  expect_error(
    sgl_parse_cox_response_cpp(
      cbind(
        c(5, 4, 3),
        c(0, 0, 0)
      )
    ),
    "at least one event"
  )
})


test_that("C++ Cox zero gradient matches R reference", {
  X <- matrix(
    c(
      1,
      0,
      0,
      1,
      2,
      1,
      1,
      2
    ),
    nrow = 4L,
    byrow = TRUE
  )

  time <- matrix(
    c(4, 3, 2, 1),
    ncol = 1L
  )

  status <- matrix(
    c(1, 0, 1, 0),
    ncol = 1L
  )

  actual <- sgl_cox_zero_gradient_cpp(
    X = X,
    time = time,
    status = status
  )

  expected <- numeric(
    ncol(X)
  )

  event_times <- sort(
    unique(
      time[status[, 1L] == 1, 1L]
    ),
    decreasing = TRUE
  )

  for (event_time in event_times) {
    event <- which(
      time[, 1L] == event_time &
        status[, 1L] == 1
    )

    risk <- which(
      time[, 1L] >= event_time
    )

    expected <- expected +
      length(event) *
        colMeans(X[risk, , drop = FALSE]) -
      colSums(X[event, , drop = FALSE])
  }

  expected <- expected /
    sum(status[, 1L] == 1)

  expect_equal(
    actual[, 1L],
    expected,
    tolerance = 1e-12
  )
})
