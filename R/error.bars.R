#' @export
error.bars <- function(x, upper, lower, width = 0.02, ...) {
  xlim <- range(x)
  barw <- diff(xlim) * width
  graphics::segments(x, upper, x, lower, ...)
  graphics::segments(x - barw, upper, x + barw, upper, ...)
  graphics::segments(x - barw, lower, x + barw, lower, ...)
  range(upper, lower)
}
