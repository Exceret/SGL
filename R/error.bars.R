#' Add error bars to an existing plot
#'
#' Draws vertical error bars centered at the points \code{x}, spanning from
#' \code{lower} to \code{upper} on the current graphics device.
#'
#' @param x Numeric vector of x coordinates at which the bars are drawn.
#' @param upper Numeric vector of upper endpoints of the bars.
#' @param lower Numeric vector of lower endpoints of the bars.
#' @param width Fraction of the x-range used for the horizontal ticks at each
#'   endpoint of the bars.  Defaults to 0.02.
#' @param ... Additional graphical parameters passed to
#'   \code{\link[graphics]{segments}}.
#'
#' @return The range of \code{c(upper, lower)}, invisibly.
#'
#' @examples
#' plot(1:5, 1:5, type = "n")
#' error.bars(1:5, upper = (1:5) + 0.5, lower = (1:5) - 0.5)
#'
#' @export
error.bars <- function(x, upper, lower, width = 0.02, ...) {
  xlim <- range(x)
  barw <- diff(xlim) * width
  graphics::segments(x, upper, x, lower, ...)
  graphics::segments(x - barw, upper, x + barw, upper, ...)
  graphics::segments(x - barw, lower, x + barw, lower, ...)
  range(upper, lower)
}