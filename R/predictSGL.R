#' Predict responses from a fitted SGL model
#'
#' Outputs predicted response values for new user input observations at a
#' specified \code{lambda} value.
#'
#' @param object A fitted \code{"SGL"} object.
#' @param newX A covariate matrix or vector for the new observations whose
#'   responses we wish to predict.
#' @param lam The index of the lambda value for the model with which we
#'   desire to predict.
#' @param ... Additional arguments are ignored.
#'
#' @return A vector of predicted responses: the linear predictor for
#'   \code{type = "linear"}, fitted probabilities for \code{type = "logit"}
#'   and the relative risk \code{exp(eta)} for \code{type = "cox"}.
#'
#' @seealso \code{\link{SGL}}
#'
#' @examples
#' set.seed(1)
#' n <- 50; p <- 100; size.groups <- 10
#' index <- ceiling(1:p / size.groups)
#' X <- matrix(rnorm(n * p), ncol = p, nrow = n)
#' beta <- (-2:2)
#' y <- X[, 1:5] %*% beta + 0.1 * rnorm(n)
#' data <- list(x = X, y = y)
#' fit <- SGL(data, index, type = "linear")
#' X.new <- matrix(rnorm(n * p), ncol = p, nrow = n)
#' pred <- predict(fit, X.new, 5)
#'
#' @export
predict.SGL <- function(object, newX, lam, ...) {
  cvobj <- object

  X <- newX

  if (is.matrix(X)) {
    X <- t(t(newX) - object$X.transform$X.means)
    if (!is.null(object$X.transform$X.scale)) {
      X <- t(t(X) / object$X.transform$X.scale)
    }
  }
  if (is.vector(X)) {
    X <- X - object$X.transform$X.means
    if (!is.null(object$X.transform$X.scale)) {
      X <- X / object$X.transform$X.scale
    }
  }

  intercept <- 0

  if (object$type == "linear") {
    intercept <- if (length(object$intercept) == 1L) {
      object$intercept
    } else {
      object$intercept[lam]
    }
  }
  if (object$type == "logit") {
    intercept <- object$intercept[lam]
  }

  if (is.matrix(X)) {
    eta <- X %*% object$beta[, lam] + intercept
  }
  if (is.vector(X)) {
    eta <- sum(X * object$beta[, lam]) + intercept
  }

  if (object$type == "linear") {
    y.pred <- eta
  }

  if (object$type == "logit") {
    y.pred <- exp(eta) / (1 + exp(eta))
  }

  if (object$type == "cox") {
    y.pred <- exp(eta)
  }

  return(y.pred)
}
