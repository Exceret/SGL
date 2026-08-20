#' Compute the negative Cox partial log-likelihood
#'
#' Computes the negative Breslow partial log-likelihood of a Cox
#' proportional hazards model for given coefficient values, using the
#' exponential of the linear predictor as the hazard multiplier.
#'
#' @param X A numeric design matrix with one row per ordered observation.
#' @param beta A numeric vector of coefficients of length \code{ncol(X)}.
#' @param death.times A numeric vector of the distinct event times.
#' @param ordered.time A numeric vector of the observed times sorted in
#'   increasing order, matching the row order of \code{X}.
#'
#' @return The negative partial log-likelihood, a single numeric value.
#'
#' @examples
#' set.seed(1)
#' n <- 30; p <- 3
#' X <- matrix(rnorm(n * p), ncol = p)
#' beta <- c(0.5, -0.2, 0.1)
#' time <- rexp(n)
#' status <- rbinom(n, 1, 0.6)
#' ordered.time <- sort(time)
#' death.times <- unique(time[status == 1])
#' log_likelihood_calc(X, beta, death.times, ordered.time)
#'
#' @export
log_likelihood_calc <- function(X, beta, death.times, ordered.time) {
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