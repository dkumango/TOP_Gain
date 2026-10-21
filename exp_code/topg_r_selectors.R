# topg_r_selectors.R
# ==================
# Runs the original R implementations of varSelRF (Diaz-Uriarte & Alvarez de Andres, 2006) and
# VSURF (Genuer, Poggi & Tuleau-Malot, 2010/2015) with their package defaults, and returns a ranked
# feature list so that the first K features can be evaluated like every other selector.
#
# Usage: Rscript topg_r_selectors.R <method: varSelRF|VSURF> <in.csv> <out.txt> <max_k> <seed> <ncores>
#   in.csv : features V1..Vp and a last column `y` (class label); no row names
#   out.txt: one 1-based feature index per line, best first (length max_k)
#
# Ranking rules (needed because both methods return a set, not a ranking):
#   varSelRF: selected variables (by initial importance), then variables in reverse order of elimination
#             (eliminated later = better; ties by initial importance), then the rest by initial importance.
#   VSURF   : prediction-step variables (in VSURF order), then remaining interpretation-step variables,
#             then remaining thresholding-step variables, then all others by mean importance (VSURF order).

args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; in_csv <- args[2]; out_txt <- args[3]
max_k <- as.integer(args[4]); seed <- as.integer(args[5]); ncores <- as.integer(args[6])
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({ library(randomForest) })

d <- read.csv(in_csv)
y <- factor(d$y)
x <- d[, setdiff(names(d), "y"), drop = FALSE]
p <- ncol(x)
set.seed(seed)
t_fit <- proc.time()[["elapsed"]]

if (method == "varSelRF") {
  suppressPackageStartupMessages(library(varSelRF))
  fit <- varSelRF(x, y, verbose = FALSE)            # package defaults (ntree 5000, ntreeIterat 2000, drop 20%)
  imp <- fit$initialImportances[, 1]
  init_order <- names(sort(imp, decreasing = TRUE))
  sel <- fit$selected.vars
  ranked <- sel[order(-imp[sel])]
  hist <- fit$selec.history$Vars.in.Forest            # strings of variables per iteration, largest set first
  sets <- lapply(as.character(hist), function(s) trimws(strsplit(s, "\\+")[[1]]))
  if (length(sets) > 1) {
    for (i in rev(seq_len(length(sets) - 1))) {      # variables dropped between iteration i and i+1
      dropped <- setdiff(sets[[i]], sets[[i + 1]])
      dropped <- setdiff(dropped, ranked)
      ranked <- c(ranked, dropped[order(-imp[dropped])])
    }
  }
  ranked <- c(ranked, setdiff(init_order, ranked))
} else if (method == "VSURF") {
  suppressPackageStartupMessages(library(VSURF))
  fit <- VSURF(x, y, parallel = (ncores > 1), ncores = ncores, clusterType = "PSOCK", verbose = FALSE)
  nm <- colnames(x)
  ranked <- nm[fit$varselect.pred]
  ranked <- c(ranked, setdiff(nm[fit$varselect.interp], ranked))
  ranked <- c(ranked, setdiff(nm[fit$varselect.thres], ranked))
  ranked <- c(ranked, setdiff(nm[fit$imp.mean.dec.ind], ranked))
  ranked <- c(ranked, setdiff(nm, ranked))
} else {
  stop(paste("unknown method", method))
}

writeLines(sprintf("%.3f", proc.time()[["elapsed"]] - t_fit), paste0(out_txt, ".time"))  # fitting time only
idx <- match(ranked, colnames(x))
writeLines(as.character(head(idx, max_k)), out_txt)
