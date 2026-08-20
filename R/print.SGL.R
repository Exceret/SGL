#' Print a summary of an SGL solution path
#'
#' Prints a short summary of a fitted \code{"SGL"} object: the regression
#' type followed by a two-column matrix of the \code{lambdas} used in the
#' regularization path and the number of non-zero coefficients at each
#' \code{lambda}.
#'
#' @param x A fitted \code{"SGL"} object.
#' @param digits Significant digits used in the printout.
#' @param ... Additional arguments passed to \code{\link[base]{print}}.
#'
#' @return The fitted object, invisibly.
#'
#' @seealso \code{\link{SGL}}
#'
#' @examples
#' set.seed(1)
#' n <- 50; p <- 100; size.groups <- 10
#' index <- ceiling(1:p / size.groups)
#' X <- matrix(rnorm(n * p), ncol = p, nrow = n)
#' y <- X[, 1:5] %*% (-2:2) + 0.1 * rnorm(n)
#' fit <- SGL(list(x = X, y = y), index, type = "linear")
#' print(fit)
#'
#' @export
print.SGL <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  num.nonzero <- apply(x$beta, 2, function(z) {
    sum(z != 0)
  })
  message("\n regression type: ", x$type, "\n\n")
  print(cbind(lambdas = x$lambdas, num.nonzero = num.nonzero))
}
