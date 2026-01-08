#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(data.table) })

ts_msg <- function(...) message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste0(..., collapse=""))

args <- commandArgs(trailingOnly=TRUE)
get_arg <- function(key, default=NULL) {
  hit <- grep(paste0("^--", key, "="), args, value=TRUE)
  if (length(hit)==0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}

INP   <- get_arg("inp",   "results/mtl_dynamic_full.tsv")
OUTD  <- get_arg("outd",  "results/causal_dynamic")
MAX_LAG <- as.integer(get_arg("max_lag", "2"))

OUTCOME_PREFIX <- "y_"

dir.create(OUTD, showWarnings=FALSE, recursive=TRUE)
if (!file.exists(INP)) stop("Missing input: ", INP)

dt <- fread(INP, showProgress=FALSE)
if (!all(c("IID","split") %in% names(dt))) stop("Need IID and split in input")

tstart_col <- c("tstart","start","time_start","t0","time_months")[c("tstart","start","time_start","t0","time_months") %in% names(dt)][1]
tstop_col  <- c("tstop","stop","time_stop","t1")[c("tstop","stop","time_stop","t1") %in% names(dt)][1]

if (is.na(tstart_col)) stop("Cannot find a start-time column (tstart/start/time_months etc.) in input")
dt[, IID := as.character(IID)]
dt[, split := tolower(trimws(as.character(split)))]
dt[, tstart := as.numeric(get(tstart_col))]
dt[, tstop  := if (!is.na(tstop_col)) as.numeric(get(tstop_col)) else NA_real_]

y_cols <- grep(paste0("^", OUTCOME_PREFIX), names(dt), value=TRUE)
if (length(y_cols) == 0) stop("No y_* outcomes found in input.")
m_cols <- sub("^y_", "m_", y_cols)
r_cols <- sub("^y_", "r_", y_cols)
have_m <- m_cols %in% names(dt)

if (!all(have_m)) {
  miss <- m_cols[!have_m]
  ts_msg("[warn] Missing mask columns: ", paste(miss, collapse=", "), " ; creating them as 1")
  for (mc in miss) dt[, (mc) := 1L]
}

# Force numeric 0/1
for (yc in y_cols) dt[, (yc) := {x <- suppressWarnings(as.numeric(get(yc))); x[is.na(x)] <- 0; as.integer(x)}]
for (mc in m_cols) dt[, (mc) := {x <- suppressWarnings(as.numeric(get(mc))); x[is.na(x)] <- 1; as.integer(x)}]

# Sort within subject to define discrete step index
setorder(dt, IID, tstart, na.last=TRUE)
dt[, step := seq_len(.N) - 1L, by=IID]

# Build "at-risk" masks per outcome:
# at_risk_y = 1 until the first event happens (based on observed y, respecting m_)
for (k in seq_along(y_cols)) {
  yc <- y_cols[k]
  mc <- m_cols[k]
  ar <- sub("^y_", "r_", yc)  # r_trait = risk mask
  # cumulative events up to previous step
  dt[, cum_prev := shift(cumsum(get(yc) * (get(mc)==1L)), n=1, fill=0L), by=IID]
  dt[, (ar) := as.integer(cum_prev == 0L)]
  dt[, cum_prev := NULL]
}

# Choose feature columns:
# drop obvious keys/labels/masks, keep everything numeric-like
drop_cols <- unique(c("FID","IID","split","interval_key","eventname","src_subject_id",
                      "tstart","tstop", tstart_col, tstop_col,
                      y_cols, m_cols, sub("^y_", "r_", y_cols)))
drop_cols <- intersect(drop_cols, names(dt))

feat_cols <- setdiff(names(dt), drop_cols)

# Coerce features to numeric where possible; keep only numeric columns
tmp <- dt[, ..feat_cols]
for (cc in names(tmp)) tmp[, (cc) := suppressWarnings(as.numeric(get(cc)))]
is_num <- vapply(tmp, function(x) is.numeric(x) || is.integer(x), logical(1))
feat_cols <- names(tmp)[is_num]
dt <- cbind(dt[, .(IID, split, tstart, tstop, step)], tmp[, ..feat_cols], dt[, ..y_cols], dt[, ..m_cols], dt[, ..r_cols])

ts_msg("Rows=", nrow(dt), " features=", length(feat_cols), " outcomes=", length(y_cols), " max_lag=", MAX_LAG)

# Create lagged columns for features and outcomes (and optionally for outcomes themselves)
make_lags <- function(D, cols, max_lag) {
  for (L in seq_len(max_lag)) {
    for (cc in cols) {
      newn <- paste0(cc, "_L", L)
      D[, (newn) := shift(get(cc), n=L, type="lag"), by=IID]
    }
  }
  D
}

# Lag features + outcomes (helps discovery and confounding control)
dt <- make_lags(dt, feat_cols, MAX_LAG)
dt <- make_lags(dt, y_cols, MAX_LAG)

# Drop first MAX_LAG rows per subject (no complete lags)
dt[, has_full_lags := as.integer(step >= 1L)]
dt2 <- dt[has_full_lags == 1L]
dt2[, has_full_lags := NULL]

# Write outputs
out_full  <- file.path(OUTD, "causal_panel_full.tsv")
out_train <- file.path(OUTD, "causal_panel_train.tsv")
out_val   <- file.path(OUTD, "causal_panel_val.tsv")
out_test  <- file.path(OUTD, "causal_panel_test.tsv")

fwrite(dt2, out_full,  sep="\t", quote=FALSE, na="NA")
fwrite(dt2[split=="train"], out_train, sep="\t", quote=FALSE, na="NA")
fwrite(dt2[split=="val"],   out_val,   sep="\t", quote=FALSE, na="NA")
fwrite(dt2[split=="test"],  out_test,  sep="\t", quote=FALSE, na="NA")

manifest <- data.table(
  param=c("INP","OUTD","MAX_LAG","N_FULL","N_TRAIN","N_VAL","N_TEST"),
  value=c(INP, OUTD, MAX_LAG, nrow(dt2), nrow(dt2[split=="train"]), nrow(dt2[split=="val"]), nrow(dt2[split=="test"]))
)
fwrite(manifest, file.path(OUTD, "build_manifest.tsv"), sep="\t")

ts_msg("DONE. Wrote: ", OUTD)

