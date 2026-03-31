# library(data.table)
# library(arrow)
# 
# # uses R-4.5.2
# # run to export/convert Cr_TrendData, DIST, STAGEDIST and WL_TrendData_2024.RData to parquet
# export_rdata <- function(
#     rdata_path,
#     out_dir,
#     format = c("parquet", "csv"),
#     max_str_level = 2
# ) {
#   format <- match.arg(format)
#   dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
#   
#   # Load into isolated environment
#   e <- new.env(parent = emptyenv())
#   load(rdata_path, envir = e)
#   
#   objs <- ls(e)
#   message("Found objects: ", paste(objs, collapse = ", "))
#   
#   for (nm in objs) {
#     obj <- e[[nm]]
#     cls <- class(obj)
#     
#     message("Processing ", nm, " (", paste(cls, collapse=", "), ")")
#     
#     # Only export tabular objects
#     if (is.data.frame(obj) || is.data.table(obj)) {
#       
#       dt <- as.data.table(obj)
#       
#       # Convert factors to character (CRITICAL for Python)
#       for (col in names(dt)) {
#         if (is.factor(dt[[col]])) {
#           dt[[col]] <- as.character(dt[[col]])
#         }
#       }
#       
#       if (format == "csv") {
#         fwrite(
#           dt,
#           file = file.path(out_dir, paste0(nm, ".csv")),
#           na = "NA"
#         )
#       }
#       
#       if (format == "parquet") {
#         write_parquet(
#           dt,
#           sink = file.path(out_dir, paste0(nm, ".parquet"))
#         )
#       }
#       
#     } else {
#       # Non-tabular objects: write structure for reference
#       txt <- capture.output(str(obj, max.level = max_str_level))
#       writeLines(
#         txt,
#         file.path(out_dir, paste0(nm, "_STRUCTURE.txt"))
#       )
#       warning("Skipped non-tabular object: ", nm)
#     }
#   }
#   
#   invisible(TRUE)
# }

# run to export/convert WLTrends.RData an CrTrends_WLlag.RData to CSV
flatten_trends <- function(trend_list) {
  nms <- names(trend_list)
  if (is.null(nms)) nms <- as.character(seq_along(trend_list))
  
  rows <- vector("list", length(trend_list))
  
  for (i in seq_along(trend_list)) {
    obj <- trend_list[[i]]
    key <- nms[i]
    
    row <- list(KEY = key)
    
    if (isS4(obj)) {
      sn <- slotNames(obj)
      
      get_scalar <- function(s) {
        if (s %in% sn) {
          v <- slot(obj, s)
          if (length(v) == 1) return(v)
        }
        NA
      }
      
      for (s in c("p_trend","COD","AIC","BIC","df","LAG")) row[[s]] <- get_scalar(s)
      
      if ("SUM" %in% sn) {
        sm <- slot(obj, "SUM")
        if (is.matrix(sm) || is.data.frame(sm)) {
          row$SUM_rows <- nrow(sm)
          row$SUM_cols <- ncol(sm)
          # optional: grab first cell as a quick sanity check
          if (nrow(sm) >= 1 && ncol(sm) >= 1) row$coef1 <- sm[1, 1]
        }
      }
      
      row$CLASS <- paste(class(obj), collapse = ",")
    } else {
      row$CLASS <- paste(class(obj), collapse = ",")
    }
    
    rows[[i]] <- row
  }
  
  # union of all column names
  all_cols <- unique(unlist(lapply(rows, names)))
  
  # fill missing columns with NA
  rows2 <- lapply(rows, function(r) {
    missing <- setdiff(all_cols, names(r))
    if (length(missing)) for (m in missing) r[[m]] <- NA
    # order consistently
    r[all_cols]
  })
  
  # bind
  df <- do.call(rbind, lapply(rows2, as.data.frame, stringsAsFactors = FALSE))
  rownames(df) <- NULL
  df
}

...length <- function(...) length(list(...))

safe_slot_scalar <- function(obj, s) {
  sn <- slotNames(obj)
  if (!(s %in% sn)) return(NA_character_)
  
  v <- slot(obj, s)
  
  if (length(v) == 0) return(NA_character_)
  
  # if matrix/data.frame/list, collapse to first atomic value or NA
  if (is.matrix(v) || is.data.frame(v)) {
    if (length(v) == 0) return(NA_character_)
    v <- v[1]
  }
  
  if (is.list(v)) {
    if (length(v) == 0) return(NA_character_)
    v <- v[[1]]
  }
  
  if (length(v) == 0) return(NA_character_)
  v <- v[1]
  
  as.character(v)
}

flatten_chlag <- function(CHLAG) {
  rows <- list()
  well_names <- names(CHLAG)
  if (is.null(well_names)) well_names <- as.character(seq_along(CHLAG))
  
  add_row <- function(key, well, iter, cls, obj=NULL) {
    list(
      KEY = as.character(key),
      WELL = as.character(well),
      ITER = as.character(iter),
      CLASS = as.character(cls),
      p_trend = if (is.null(obj)) NA_character_ else safe_slot_scalar(obj, "p_trend"),
      AIC = if (is.null(obj)) NA_character_ else safe_slot_scalar(obj, "AIC"),
      BIC = if (is.null(obj)) NA_character_ else safe_slot_scalar(obj, "BIC"),
      LAG = if (is.null(obj)) NA_character_ else safe_slot_scalar(obj, "LAG")
    )
  }
  
  for (i in seq_along(CHLAG)) {
    nm <- well_names[i]
    obj <- CHLAG[[i]]
    
    if (isS4(obj)) {
      rows[[length(rows) + 1]] <- add_row(
        key = paste0(nm, "__ITER1"),
        well = nm,
        iter = 1,
        cls = paste(class(obj), collapse = ","),
        obj = obj
      )
    } else if (is.list(obj)) {
      subn <- names(obj)
      if (is.null(subn)) subn <- paste0("ITER", seq_along(obj))
      
      for (j in seq_along(obj)) {
        sub <- obj[[j]]
        iter <- suppressWarnings(as.integer(gsub("ITER", "", subn[j])))
        if (is.na(iter)) iter <- j
        
        if (isS4(sub)) {
          rows[[length(rows) + 1]] <- add_row(
            key = paste0(nm, "__ITER", iter),
            well = nm,
            iter = iter,
            cls = paste(class(sub), collapse = ","),
            obj = sub
          )
        } else {
          rows[[length(rows) + 1]] <- add_row(
            key = paste0(nm, "__ITER", iter),
            well = nm,
            iter = iter,
            cls = paste(class(sub), collapse = ","),
            obj = NULL
          )
        }
      }
    } else {
      rows[[length(rows) + 1]] <- add_row(
        key = paste0(nm, "__ITER1"),
        well = nm,
        iter = 1,
        cls = paste(class(obj), collapse = ","),
        obj = NULL
      )
    }
  }
  
  # bind manually, no rbind guessing
  df <- data.frame(
    KEY = sapply(rows, `[[`, "KEY"),
    WELL = sapply(rows, `[[`, "WELL"),
    ITER = sapply(rows, `[[`, "ITER"),
    CLASS = sapply(rows, `[[`, "CLASS"),
    p_trend = sapply(rows, `[[`, "p_trend"),
    AIC = sapply(rows, `[[`, "AIC"),
    BIC = sapply(rows, `[[`, "BIC"),
    LAG = sapply(rows, `[[`, "LAG"),
    stringsAsFactors = FALSE
  )
  
  df
}

# ============================================
# Exporter for CHLAG / CrTrends
# ============================================


safe_trim <- function(x) {
  x <- as.character(x)
  gsub("^\\s+|\\s+$", "", x)
}

as_scalar_chr <- function(x) {
  if (is.null(x)) return(NA_character_)
  if (length(x) == 0) return(NA_character_)
  
  if (inherits(x, "formula")) {
    return(paste(deparse(x), collapse = " "))
  }
  
  if (is.matrix(x) || is.data.frame(x)) {
    if (length(x) == 0) return(NA_character_)
    x <- x[1]
  } else if (is.list(x)) {
    if (length(x) == 0) return(NA_character_)
    x <- x[[1]]
  }
  
  if (length(x) == 0) return(NA_character_)
  as.character(x[1])
}

as_scalar_num <- function(x) {
  suppressWarnings(as.numeric(as_scalar_chr(x)))
}

safe_slot_exists <- function(obj, s) {
  isS4(obj) && (s %in% methods::slotNames(obj))
}

safe_slot <- function(obj, s, default = NULL) {
  if (!isS4(obj)) return(default)
  sn <- methods::slotNames(obj)
  if (!(s %in% sn)) return(default)
  methods::slot(obj, s)
}

safe_slot_raw <- function(obj, s) {
  if (!safe_slot_exists(obj, s)) return(NA_character_)
  as_scalar_chr(methods::slot(obj, s))
}

safe_slot_num <- function(obj, s) {
  suppressWarnings(as.numeric(safe_slot_raw(obj, s)))
}

normalize_formula_label <- function(x) {
  x <- safe_trim(x)
  x <- gsub("log\\(VAL\\)\\s*~\\s*", "", x)
  x <- gsub("\\s+", " ", x)
  x
}

extract_ic_fields <- function(obj) {
  out <- list(
    FORM_raw = NA_character_,
    FORM_label = NA_character_,
    AIC_full = NA_real_,
    BIC_full = NA_real_,
    RL_full = NA_real_,
    AIC_EVENT = NA_real_,
    AIC_INTERP = NA_real_,
    AIC_NULL = NA_real_,
    BIC_EVENT = NA_real_,
    BIC_INTERP = NA_real_,
    BIC_NULL = NA_real_
  )
  
  if (!isS4(obj)) return(out)
  
  # formula slot
  if (safe_slot_exists(obj, "FORM")) {
    form_raw <- methods::slot(obj, "FORM")
    out$FORM_raw <- as_scalar_chr(form_raw)
    out$FORM_label <- normalize_formula_label(out$FORM_raw)
  }
  
  # AIC table
  if (safe_slot_exists(obj, "AIC")) {
    aic_tab <- methods::slot(obj, "AIC")
    if (is.data.frame(aic_tab) || is.matrix(aic_tab)) {
      aic_tab <- as.data.frame(aic_tab, stringsAsFactors = FALSE)
      names(aic_tab) <- safe_trim(names(aic_tab))
      
      if (all(c("Formula", "AIC") %in% names(aic_tab))) {
        aic_tab$Formula_norm <- normalize_formula_label(aic_tab$Formula)
        
        # full model row = row matching FORM slot
        idx <- which(aic_tab$Formula_norm == out$FORM_label)
        if (length(idx) > 0) {
          out$AIC_full <- suppressWarnings(as.numeric(aic_tab$AIC[idx[1]]))
          if ("RL" %in% names(aic_tab)) {
            out$RL_full <- suppressWarnings(as.numeric(aic_tab$RL[idx[1]]))
          }
        }
        
        # candidate rows
        idx0 <- which(aic_tab$Formula_norm == "0")
        idxE <- which(aic_tab$Formula_norm == "EVENT")
        idxI <- which(aic_tab$Formula_norm == "INTERP")
        
        if (length(idx0) > 0) out$AIC_NULL   <- suppressWarnings(as.numeric(aic_tab$AIC[idx0[1]]))
        if (length(idxE) > 0) out$AIC_EVENT  <- suppressWarnings(as.numeric(aic_tab$AIC[idxE[1]]))
        if (length(idxI) > 0) out$AIC_INTERP <- suppressWarnings(as.numeric(aic_tab$AIC[idxI[1]]))
      }
    }
  }
  
  # BIC table
  if (safe_slot_exists(obj, "BIC")) {
    bic_tab <- methods::slot(obj, "BIC")
    if (is.data.frame(bic_tab) || is.matrix(bic_tab)) {
      bic_tab <- as.data.frame(bic_tab, stringsAsFactors = FALSE)
      names(bic_tab) <- safe_trim(names(bic_tab))
      
      if (all(c("Formula", "BIC") %in% names(bic_tab))) {
        bic_tab$Formula_norm <- normalize_formula_label(bic_tab$Formula)
        
        idx <- which(bic_tab$Formula_norm == out$FORM_label)
        if (length(idx) > 0) {
          out$BIC_full <- suppressWarnings(as.numeric(bic_tab$BIC[idx[1]]))
        }
        
        idx0 <- which(bic_tab$Formula_norm == "0")
        idxE <- which(bic_tab$Formula_norm == "EVENT")
        idxI <- which(bic_tab$Formula_norm == "INTERP")
        
        if (length(idx0) > 0) out$BIC_NULL   <- suppressWarnings(as.numeric(bic_tab$BIC[idx0[1]]))
        if (length(idxE) > 0) out$BIC_EVENT  <- suppressWarnings(as.numeric(bic_tab$BIC[idxE[1]]))
        if (length(idxI) > 0) out$BIC_INTERP <- suppressWarnings(as.numeric(bic_tab$BIC[idxI[1]]))
      }
    }
  }
  
  out
}

extract_sum_fields_v3 <- function(obj) {
  out <- list(
    SUM_rows = NA_real_,
    SUM_cols = NA_real_,
    SUM_row_names = NA_character_,
    SUM_col_names = NA_character_,
    beta_intercept = NA_real_,
    beta_interp = NA_real_,
    beta_event = NA_real_,
    beta_logSigma = NA_real_,
    se_intercept = NA_real_,
    se_interp = NA_real_,
    se_event = NA_real_,
    se_logSigma = NA_real_,
    p_interp = NA_real_,
    p_event = NA_real_,
    n_obs = NA_real_
  )
  
  if (!safe_slot_exists(obj, "SUM")) return(out)
  
  sm <- methods::slot(obj, "SUM")
  if (is.null(sm)) return(out)
  if (!(is.matrix(sm) || is.data.frame(sm))) return(out)
  
  sm <- as.data.frame(sm, stringsAsFactors = FALSE)
  out$SUM_rows <- nrow(sm)
  out$SUM_cols <- ncol(sm)
  out$n_obs <- nrow(safe_slot(obj, "PRED", default = numeric(0)))
  
  rn <- rownames(sm)
  cn <- colnames(sm)
  
  rn_low <- if (is.null(rn)) character(0) else tolower(safe_trim(rn))
  cn_low <- if (is.null(cn)) character(0) else tolower(safe_trim(cn))
  
  out$SUM_row_names <- if (length(rn)) paste(rn, collapse = " | ") else NA_character_
  out$SUM_col_names <- if (length(cn)) paste(cn, collapse = " | ") else NA_character_
  
  get_cell <- function(row_candidates, col_candidates) {
    if (length(rn_low) == 0 || length(cn_low) == 0) return(NA_real_)
    r_idx <- match(tolower(row_candidates), rn_low, nomatch = 0)
    c_idx <- match(tolower(col_candidates), cn_low, nomatch = 0)
    r_idx <- r_idx[r_idx > 0]
    c_idx <- c_idx[c_idx > 0]
    if (length(r_idx) == 0 || length(c_idx) == 0) return(NA_real_)
    suppressWarnings(as.numeric(sm[r_idx[1], c_idx[1]]))
  }
  
  est_cols <- c("estimate", "est", "coef", "coefficient", "value")
  se_cols  <- c("std. error", "std.error", "se", "stderr", "s.e.")
  p_cols   <- c("pr(> t)", "pr(>|t|)", "pr(>|z|)", "p.value", "p-value", "p")
  
  out$beta_intercept <- get_cell(c("(intercept)", "intercept"), est_cols)
  out$beta_interp    <- get_cell(c("interp", "interplag"), est_cols)
  out$beta_event     <- get_cell(c("event", "time"), est_cols)
  out$beta_logSigma  <- get_cell(c("logsigma"), est_cols)
  
  out$se_intercept   <- get_cell(c("(intercept)", "intercept"), se_cols)
  out$se_interp      <- get_cell(c("interp", "interplag"), se_cols)
  out$se_event       <- get_cell(c("event", "time"), se_cols)
  out$se_logSigma    <- get_cell(c("logsigma"), se_cols)
  
  out$p_interp       <- get_cell(c("interp", "interplag"), p_cols)
  out$p_event        <- get_cell(c("event", "time"), p_cols)
  
  out
}

extract_one_trend_row_v6 <- function(obj, well, iter) {
  cls <- paste(class(obj), collapse = ",")
  
  if (!isS4(obj)) {
    row <- list(
      KEY = paste0(well, "__ITER", iter),
      WELL = as.character(well),
      ITER = as.integer(iter),
      CLASS = cls,
      MODEL = NA_character_,
      FORM_raw = NA_character_,
      FORM_label = NA_character_,
      LOG = NA_character_,
      LAG = NA_real_,
      p_trend = NA_real_,
      df = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_,
      RL = NA_real_,
      AIC_EVENT = NA_real_,
      AIC_INTERP = NA_real_,
      AIC_NULL = NA_real_,
      BIC_EVENT = NA_real_,
      BIC_INTERP = NA_real_,
      BIC_NULL = NA_real_,
      logLik = NA_real_,
      n_obs = NA_real_,
      beta_intercept = NA_real_,
      beta_interp = NA_real_,
      beta_event = NA_real_,
      beta_logSigma = NA_real_,
      p_interp = NA_real_,
      p_event = NA_real_,
      SUM_rows = NA_real_,
      SUM_cols = NA_real_,
      model_type = "NON_S4",
      fit_ok = FALSE
    )
  } else {
    icf  <- extract_ic_fields(obj)
    sumf <- extract_sum_fields_v3(obj)
    
    model_slot <- safe_slot_raw(obj, "MODEL")
    log_slot   <- safe_slot_raw(obj, "LOG")
    
    model_type <- "UNKNOWN"
    if (!is.na(sumf$beta_interp) && !is.na(sumf$beta_event)) {
      model_type <- "INTERP+EVENT"
    } else if (is.na(sumf$beta_interp) && !is.na(sumf$beta_event)) {
      model_type <- "EVENT"
    } else if (!is.na(sumf$SUM_rows)) {
      model_type <- paste0("SUM_ROWS_", as.integer(sumf$SUM_rows))
    }
    
    fit_ok <- any(!is.na(c(
      safe_slot_num(obj, "p_trend"),
      icf$AIC_full,
      icf$BIC_full,
      sumf$beta_event,
      sumf$beta_interp
    )))
    
    row <- list(
      KEY = paste0(well, "__ITER", iter),
      WELL = as.character(well),
      ITER = as.integer(iter),
      CLASS = cls,
      MODEL = model_slot,
      FORM_raw = icf$FORM_raw,
      FORM_label = icf$FORM_label,
      LOG = log_slot,
      LAG = safe_slot_num(obj, "LAG"),
      p_trend = safe_slot_num(obj, "p_trend"),
      df = safe_slot_num(obj, "df"),
      
      AIC = icf$AIC_full,
      BIC = icf$BIC_full,
      RL = icf$RL_full,
      AIC_EVENT = icf$AIC_EVENT,
      AIC_INTERP = icf$AIC_INTERP,
      AIC_NULL = icf$AIC_NULL,
      BIC_EVENT = icf$BIC_EVENT,
      BIC_INTERP = icf$BIC_INTERP,
      BIC_NULL = icf$BIC_NULL,
      
      logLik = NA_real_,
      n_obs = sumf$n_obs,
      
      beta_intercept = sumf$beta_intercept,
      beta_interp = sumf$beta_interp,
      beta_event = sumf$beta_event,
      beta_logSigma = sumf$beta_logSigma,
      p_interp = sumf$p_interp,
      p_event = sumf$p_event,
      
      SUM_rows = sumf$SUM_rows,
      SUM_cols = sumf$SUM_cols,
      model_type = model_type,
      fit_ok = fit_ok
    )
  }
  
  # hard-normalize every field to length 1
  row <- lapply(row, function(x) {
    if (is.null(x) || length(x) == 0) return(NA)
    if (inherits(x, "formula")) return(paste(deparse(x), collapse = " "))
    if (is.matrix(x) || is.data.frame(x)) x <- x[1]
    else if (is.list(x)) x <- x[[1]]
    if (length(x) == 0) return(NA)
    x[1]
  })
  
  # build 1-row data.frame safely
  out <- as.data.frame(row, stringsAsFactors = FALSE)
  
  # type-fix selected columns
  num_cols <- c(
    "ITER","LAG","p_trend","df","AIC","BIC","RL",
    "AIC_EVENT","AIC_INTERP","AIC_NULL",
    "BIC_EVENT","BIC_INTERP","BIC_NULL",
    "logLik","n_obs",
    "beta_intercept","beta_interp","beta_event","beta_logSigma",
    "p_interp","p_event","SUM_rows","SUM_cols"
  )
  for (nm in intersect(num_cols, names(out))) {
    out[[nm]] <- suppressWarnings(as.numeric(out[[nm]]))
  }
  if ("fit_ok" %in% names(out)) out$fit_ok <- as.logical(out$fit_ok)
  
  out
}

flatten_crtrends_complete_v6 <- function(CHLAG) {
  rows <- list()
  well_names <- names(CHLAG)
  if (is.null(well_names)) well_names <- as.character(seq_along(CHLAG))
  
  push_row <- function(obj, well, iter) {
    out <- tryCatch(
      extract_one_trend_row_v6(obj, well, iter),
      error = function(e) {
        message("FAILED on well=", well, " iter=", iter, " class=", paste(class(obj), collapse = ","))
        stop(e)
      }
    )
    rows[[length(rows) + 1]] <<- out
  }
  
  for (i in seq_along(CHLAG)) {
    nm <- well_names[i]
    obj <- CHLAG[[i]]
    
    if (isS4(obj)) {
      push_row(obj, nm, 1L)
    } else if (is.list(obj)) {
      subn <- names(obj)
      if (is.null(subn)) subn <- paste0("ITER", seq_along(obj))
      for (j in seq_along(obj)) {
        iter <- suppressWarnings(as.integer(gsub("^ITER", "", subn[j], ignore.case = TRUE)))
        if (is.na(iter)) iter <- j
        push_row(obj[[j]], nm, iter)
      }
    } else {
      push_row(obj, nm, 1L)
    }
  }
  
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out <- out[order(out$WELL, out$ITER), , drop = FALSE]
  rownames(out) <- NULL
  out
}