SGL_lambda_max_from_gradient <- function(
  gradient,
  group_index,
  group_weight,
  alpha
) {
  sgl_lambda_max_from_gradient_cpp(
    gradient = matrix(
      as.numeric(gradient),
      ncol = 1L
    ),
    group_index = matrix(
      as.numeric(group_index),
      ncol = 1L
    ),
    group_weight = matrix(
      as.numeric(group_weight),
      ncol = 1L
    ),
    alpha = as.numeric(alpha)
  )
}


SGL_make_lambda_path <- function(
  lambdas,
  lambda_max,
  nlam,
  min_frac,
  gamma
) {
  sgl_make_lambda_path_cpp(
    lambda_input = lambdas,
    lambda_max = as.numeric(lambda_max),
    n_lambda = as.integer(nlam),
    min_frac = as.numeric(min_frac),
    lambda_decay = as.numeric(gamma)
  )
}


SGL_cox_zero_gradient <- function(
  X,
  time,
  status
) {
  sgl_cox_zero_gradient_cpp(
    X = X,
    time = matrix(
      as.numeric(time),
      ncol = 1L
    ),
    status = matrix(
      as.numeric(status),
      ncol = 1L
    )
  )[, 1L]
}

SGL_transform_train_test <- function(
  X_train,
  X_test,
  standardize = TRUE
) {
  transformed <- sgl_center_scale_cpp(
    X = X_train,
    standardize = isTRUE(standardize)
  )

  X_train_transformed <- transformed$x

  X_test_transformed <- sgl_apply_center_scale_cpp(
    X = X_test,
    X_transform = transformed$X.transform,
    standardize = isTRUE(standardize)
  )

  list(
    x_train = X_train_transformed,
    x_test = X_test_transformed,
    X.transform = transformed$X.transform
  )
}
