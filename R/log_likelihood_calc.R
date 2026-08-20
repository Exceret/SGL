#' @export
log.likelihood.calc <- function(X, beta, death.times, ordered.time) {
  eta <- drop(X %*% beta)
  ## match ordered.time to death.times (NA for censor-only times)
  mt <- match(ordered.time, death.times)
  ## number of observations at each death time (tabulate ignores NA)
  nd <- tabulate(mt, nbins = length(death.times))
  ## sum of eta over the block at each death time
  numer <- tapply(eta, mt, sum)
  ## risk set = suffix starting at the first occurrence of each death time
  rs.start <- match(death.times, ordered.time)
  suff <- rev(cumsum(rev(exp(eta))))
  denom <- suff[rs.start]^nd
  return(-sum(numer - log(denom)))
}
