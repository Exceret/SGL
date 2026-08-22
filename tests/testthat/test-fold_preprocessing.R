test_that("validation data does not affect fold preprocessing", {
  X_train <- matrix(
    c(
      1,
      10,
      2,
      20,
      3,
      30
    ),
    ncol = 2,
    byrow = TRUE
  )

  X_test_1 <- matrix(
    c(100, 1000),
    nrow = 1
  )

  X_test_2 <- matrix(
    c(10000, 100000),
    nrow = 1
  )

  result_1 <- SGL2:::SGL_transform_train_test(
    X_train = X_train,
    X_test = X_test_1,
    standardize = TRUE
  )

  result_2 <- SGL2:::SGL_transform_train_test(
    X_train = X_train,
    X_test = X_test_2,
    standardize = TRUE
  )

  # 训练集变换不能受验证集取值影响
  expect_equal(
    result_1$x_train,
    result_2$x_train
  )

  # 验证集本身不同，变换后的值应不同
  expect_false(
    isTRUE(
      all.equal(
        result_1$x_test,
        result_2$x_test
      )
    )
  )

  # 训练集均值和 scale 应来自训练集
  expected <- sgl_center_scale_cpp(
    X_train,
    TRUE
  )

  expect_equal(
    result_1$x_train,
    expected$x
  )
})
