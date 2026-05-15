#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
infile <- args[1]
outfile <- args[2]

dat <- read.csv(infile)

x1 <- dat$x1[!is.na(dat$x1) & !is.na(dat$y1)]
y1 <- dat$y1[!is.na(dat$x1) & !is.na(dat$y1)]
x2 <- dat$x2[!is.na(dat$x2) & !is.na(dat$y2)]
y2 <- dat$y2[!is.na(dat$x2) & !is.na(dat$y2)]
lag <- dat$lag[1]

crosscor_exact <- function(x1, y1, x2, y2, lag = 0) {
  lag <- round(lag)
  x.lag <- x1 - lag
  x <- intersect(x2, x.lag)
  i1 <- sapply(x + lag, function(u) which.max(u == x1))
  i2 <- sapply(x, function(u) which.max(u == x2))
  x1 <- x1[i1]
  y1 <- y1[i1]
  x2 <- x2[i2]
  y2 <- y2[i2]
  r1 <- residuals(loess(y1 ~ x1))
  r2 <- residuals(loess(y2 ~ x2))
  Xts <- ts.intersect(as.ts(r1), as.ts(r2))
  colnames(Xts) <- c("r1", "r2")
  acf.out <- acf(Xts, lag.max = 0, plot = FALSE, type = "correlation", na.action = na.fail)
  y <- c(rev(acf.out$acf[-1, 2, 1]), acf.out$acf[, 1, 2])
  data.frame(acf = y, lag = lag)
}

out <- crosscor_exact(x1, y1, x2, y2, lag)
write.csv(out, outfile, row.names = FALSE)