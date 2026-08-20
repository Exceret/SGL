test_that("SGL is a compatibility adapter over sgl_fit", {
  set.seed(20250320)

  n <- 60L
  p <- 8L

  X <- matrix(
    rnorm(n * p),
    nrow = n,
    ncol = p
  )

  y <- matrix(
    rnorm(n),
    ncol = 1L
  )

  index <- 1:8

  lambdas <- c(
    0.2,
    0.1,
    0.05
  )

  legacy <- SGL::SGL(
    data = list(
      x = X,
      y = y
    ),
    index = index,
    type = "linear",
    standardize = TRUE,
    maxit = 1000L,
    thresh = 1e-8,
    lambdas = lambdas,
    alpha = 0.95
  )

  direct <- SGL(
    data = list(
      x = X,
      y = y
    ),
    index = index,
    type = "linear",
    standardize = TRUE,
    maxit = 1000L,
    thresh = 1e-8,
    lambdas = lambdas,
    alpha = 0.95
  )

  expect_identical(
    legacy$type,
    "linear"
  )

  expect_identical(
    legacy$lambdas,
    direct$lambdas
  )

  expect_lt(
    max(abs(
      legacy$beta -
        direct$beta
    )),
    1e-10
  )

  expect_equal(
    legacy$intercept,
    direct$intercept,
    tolerance = 1e-12
  )

  expect_identical(
    legacy$X.transform,
    direct$X.transform
  )
})
