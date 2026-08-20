#' Predict responses from a fitted SGL model
#'
#' Outputs predicted response values for new user input observations at a
#' specified \code{lambda} value.
#'
#' @param x A fitted \code{"SGL"} object.
#' @param newX A covariate matrix or vector for the new observations whose
#'   responses we wish to predict.
#' @param lam The index of the lambda value for the model with which we
#'   desire to predict.
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
#' pred <- predictSGL(fit, X.new, 5)
#'
#' @export
predictSGL <- function(x, newX, lam) {
  cvobj <- x

  X <- newX

  if (is.matrix(X)) {
    X <- t(t(newX) - x$X.transform$X.means)
    if (!is.null(x$X.transform$X.scale)) {
      X <- t(t(X) / x$X.transform$X.scale)
    }
  }
  if (is.vector(X)) {
    X <- X - x$X.transform$X.means
    if (!is.null(x$X.transform$X.scale)) {
      X <- X / x$X.transform$X.scale
    }
  }

  intercept <- 0

  if (x$type == "linear") {
    intercept <- x$intercept
  }
  if (x$type == "logit") {
    intercept <- x$intercept[lam]
  }

  if (is.matrix(X)) {
    eta <- X %*% x$beta[, lam] + intercept
  }
  if (is.vector(X)) {
    eta <- sum(X * x$beta[, lam]) + intercept
  }

  if (x$type == "linear") {
    y.pred <- eta
  }

  if (x$type == "logit") {
    y.pred <- exp(eta) / (1 + exp(eta))
  }

  if (x$type == "cox") {
    y.pred <- exp(eta)
  }

  return(y.pred)
}