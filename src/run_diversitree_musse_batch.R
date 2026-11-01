#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ape)
  library(diversitree)
})

args <- commandArgs(trailingOnly = TRUE)
batch_dir <- if (length(args) >= 1) {
  normalizePath(args[[1]], mustWork = TRUE)
} else {
  normalizePath(
    "data/simulation/coord_diff/star_2/setting_2/batch_0",
    mustWork = TRUE
  )
}
output_dir <- if (length(args) >= 2) {
  args[[2]]
} else {
  file.path(batch_dir, "diversitree_musse")
}
maxit <- if (length(args) >= 3) as.integer(args[[3]]) else 500L
if (is.na(maxit) || maxit < 1L) {
  stop("maxit must be a positive integer")
}
resume <- length(args) >= 4 && identical(args[[4]], "--resume")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

tree_dirs <- list.dirs(batch_dir, recursive = FALSE, full.names = TRUE)
tree_dirs <- tree_dirs[grepl("/tree_[0-9]+$", tree_dirs)]
has_nodes <- file.exists(file.path(tree_dirs, "nodes.csv"))
has_edges <- file.exists(file.path(tree_dirs, "edges.csv"))
if (any(xor(has_nodes, has_edges))) {
  stop("Found a tree directory with only one of nodes.csv and edges.csv")
}
tree_dirs <- tree_dirs[has_nodes & has_edges]
if (!length(tree_dirs)) {
  stop("No tree directories found in ", batch_dir)
}
tree_tip_counts <- vapply(
  tree_dirs,
  function(tree_dir) {
    nodes <- read.csv(
      file.path(tree_dir, "nodes.csv"),
      stringsAsFactors = FALSE
    )
    if (!"is_leaf" %in% names(nodes) ||
        anyNA(nodes$is_leaf) ||
        any(!nodes$is_leaf %in% c(0, 1))) {
      stop("Invalid is_leaf values in ", file.path(tree_dir, "nodes.csv"))
    }
    as.integer(sum(nodes$is_leaf == 1))
  },
  integer(1)
)
if (any(tree_tip_counts == 0L)) {
  stop("Found a tree with no sampled tips in ", batch_dir)
}
excluded_single_tip_dirs <- tree_dirs[tree_tip_counts == 1L]
excluded_single_tip_trees <- data.frame(
  tree = sub("^.*tree_([0-9]+)$", "\\1", excluded_single_tip_dirs),
  tip_count = tree_tip_counts[tree_tip_counts == 1L],
  reason = rep("single_sampled_tip", length(excluded_single_tip_dirs)),
  stringsAsFactors = FALSE
)
if (nrow(excluded_single_tip_trees)) {
  cat(
    "Excluding single-tip trees:",
    paste(excluded_single_tip_trees$tree, collapse = ","),
    "\n"
  )
  unlink(file.path(
    output_dir,
    paste0("tree_", excluded_single_tip_trees$tree, ".nwk")
  ))
  unlink(file.path(
    output_dir,
    paste0("tree_", excluded_single_tip_trees$tree, "_states.csv")
  ))
}
tree_dirs <- tree_dirs[tree_tip_counts >= 2L]
if (!length(tree_dirs)) {
  stop("No trees with at least two sampled tips in ", batch_dir)
}
tree_indices <- as.integer(sub("^.*tree_([0-9]+)$", "\\1", tree_dirs))
tree_dirs <- tree_dirs[order(tree_indices)]
write.csv(
  excluded_single_tip_trees,
  file.path(output_dir, "excluded_trees.csv"),
  row.names = FALSE
)

read_tree <- function(tree_dir) {
  nodes_path <- file.path(tree_dir, "nodes.csv")
  edges_path <- file.path(tree_dir, "edges.csv")
  if (!file.exists(nodes_path) || !file.exists(edges_path)) {
    stop("Missing nodes.csv or edges.csv in ", tree_dir)
  }

  nodes <- read.csv(nodes_path, stringsAsFactors = FALSE)
  edges <- read.csv(edges_path, stringsAsFactors = FALSE)
  required_nodes <- c("node_id", "is_leaf", "state_id")
  required_edges <- c("parent", "child", "branch_length")
  if (!all(required_nodes %in% names(nodes))) {
    stop("Invalid node columns in ", nodes_path)
  }
  if (!all(required_edges %in% names(edges))) {
    stop("Invalid edge columns in ", edges_path)
  }
  if (anyDuplicated(nodes$node_id) || anyDuplicated(edges$child)) {
    stop("Node IDs must be unique and each node may have only one parent in ", tree_dir)
  }
  if (any(!is.finite(edges$branch_length)) || any(edges$branch_length < 0)) {
    stop("Branch lengths must be finite and non-negative in ", edges_path)
  }

  root_ids <- setdiff(nodes$node_id, edges$child)
  if (length(root_ids) != 1L) {
    stop("Expected one root node in ", tree_dir)
  }
  root_id <- root_ids[[1]]
  root_edges <- which(edges$parent == root_id)
  if (length(root_edges) != 1L) {
    stop("Expected exactly one stem edge at the root in ", tree_dir)
  }
  stem_length <- edges$branch_length[root_edges]
  children_by_parent <- split(seq_len(nrow(edges)), edges$parent)
  visited <- character()

  make_newick_subtree <- function(node_id) {
    node_key <- as.character(node_id)
    if (node_key %in% visited) {
      stop("Cycle detected in ", tree_dir)
    }
    visited <<- c(visited, node_key)
    child_rows <- children_by_parent[[node_key]]
    if (is.null(child_rows)) {
      return(node_key)
    }

    descendants <- vapply(
      child_rows,
      function(row) {
        paste0(
          make_newick_subtree(edges$child[row]),
          ":",
          sprintf("%.17g", edges$branch_length[row])
        )
      },
      character(1)
    )
    paste0("(", paste(descendants, collapse = ","), ")")
  }

  newick <- paste0(make_newick_subtree(root_id), ";")
  if (length(visited) != nrow(nodes)) {
    stop("Tree contains disconnected nodes in ", tree_dir)
  }
  tree <- ape::read.tree(text = newick)
  tree <- ape::collapse.singles(tree, root.edge = FALSE)
  if (ape::has.singles(tree)) {
    stop("Singleton nodes remain after dropping the stem in ", tree_dir)
  }
  if (!ape::is.binary(tree)) {
    stop("Expected a fully bifurcating tree after dropping the stem in ", tree_dir)
  }
  tip_depths <- ape::node.depth.edgelength(tree)[seq_len(ape::Ntip(tree))]
  if (max(tip_depths) - min(tip_depths) > 1e-8) {
    stop("Tree is not ultrametric after dropping the stem in ", tree_dir)
  }
  if (!is.null(tree$root.edge)) {
    stop("Unexpected root.edge after dropping the stem in ", tree_dir)
  }

  tip_rows <- nodes[match(tree$tip.label, as.character(nodes$node_id)), ]
  if (anyNA(tip_rows$node_id) || any(tip_rows$is_leaf != 1)) {
    stop("Could not match every phylogeny tip to a sampled node in ", tree_dir)
  }
  if (any(!tip_rows$state_id %in% 0:3)) {
    stop("Tip state IDs must be in 0:3 in ", tree_dir)
  }
  states <- setNames(as.integer(tip_rows$state_id) + 1L, tree$tip.label)

  tree_index <- sub("^.*tree_([0-9]+)$", "\\1", tree_dir)
  ape::write.tree(
    tree,
    file = file.path(output_dir, paste0("tree_", tree_index, ".nwk"))
  )
  write.csv(
    data.frame(
      tip_label = names(states),
      state_id = as.integer(states) - 1L,
      state = c("A", "B", "C", "D")[as.integer(states)],
      stringsAsFactors = FALSE
    ),
    file.path(output_dir, paste0("tree_", tree_index, "_states.csv")),
    row.names = FALSE
  )

  list(
    tree = tree,
    states = states,
    tree_index = tree_index,
    stem_length = stem_length,
    tip_counts = tabulate(as.integer(states), nbins = 4L)
  )
}

tree_data <- lapply(tree_dirs, read_tree)
names(tree_data) <- vapply(tree_data, `[[`, character(1), "tree_index")

sampling_fraction <- rep(0.5, 4L)
base_likelihoods <- lapply(
  tree_data,
  function(item) {
    diversitree::make.musse(
      tree = item$tree,
      states = item$states,
      k = 4,
      sampling.f = sampling_fraction,
      strict = FALSE
    )
  }
)
constrained_likelihoods <- lapply(
  base_likelihoods,
  function(likelihood) {
    diversitree::constrain(
      likelihood,
      q21 ~ 0,
      q23 ~ 0,
      q24 ~ 0,
      q31 ~ 0,
      q32 ~ 0,
      q34 ~ 0,
      q41 ~ 0,
      q42 ~ 0,
      q43 ~ 0
    )
  }
)

parameter_names <- diversitree::argnames(constrained_likelihoods[[1]])
starting_values <- diversitree::starting.point.musse(
  tree_data[[1]]$tree,
  k = 4
)[parameter_names]
estimates_path <- file.path(output_dir, "parameter_estimates.csv")
if (resume && file.exists(estimates_path)) {
  previous_estimates <- read.csv(estimates_path, stringsAsFactors = FALSE)
  if (!all(c("parameter", "estimate") %in% names(previous_estimates)) ||
      anyDuplicated(previous_estimates$parameter) ||
      !setequal(previous_estimates$parameter, parameter_names)) {
    stop("Existing parameter estimates do not match the current MuSSE model")
  }
  previous_values <- setNames(
    previous_estimates$estimate,
    previous_estimates$parameter
  )
  starting_values <- previous_values[parameter_names]
  cat("Resuming optimization from saved parameter estimates\n")
}
if (anyNA(starting_values) || any(!is.finite(starting_values))) {
  stop("Could not construct finite initial values for all free MuSSE parameters")
}

tree_log_likelihoods <- function(parameters) {
  vapply(
    constrained_likelihoods,
    function(likelihood) {
      as.numeric(
        likelihood(
          parameters,
          condition.surv = TRUE,
          root = diversitree::ROOT.OBS
        )
      )
    },
    numeric(1)
  )
}
joint_log_likelihood <- function(parameters) {
  contributions <- tree_log_likelihoods(parameters)
  if (any(!is.finite(contributions))) {
    return(-Inf)
  }
  sum(contributions)
}
objective <- function(parameters) {
  likelihood <- joint_log_likelihood(parameters)
  if (!is.finite(likelihood)) {
    return(.Machine$double.xmax^0.25)
  }
  -likelihood
}

initial_log_likelihood <- joint_log_likelihood(starting_values)
if (!is.finite(initial_log_likelihood)) {
  stop("Initial joint MuSSE log likelihood is not finite")
}
cat("Trees:", length(tree_data), "\n")
cat("Pooled tip counts A,B,C,D:",
    paste(Reduce(`+`, lapply(tree_data, `[[`, "tip_counts")), collapse = ","),
    "\n")
cat("Initial joint log likelihood:", initial_log_likelihood, "\n")
cat("Starting optimization with", length(parameter_names), "free parameters\n")

fit <- stats::optim(
  par = starting_values,
  fn = objective,
  method = "L-BFGS-B",
  lower = rep(0, length(starting_values)),
  upper = rep(Inf, length(starting_values)),
  control = list(maxit = maxit, trace = 1, REPORT = 10)
)
final_log_likelihood <- joint_log_likelihood(fit$par)
if (!is.finite(final_log_likelihood)) {
  stop("Optimization ended with a non-finite joint likelihood")
}
writeLines(
  c(
    paste0("optimizer=L-BFGS-B; maxit=", maxit),
    paste0("convergence_code=", fit$convergence),
    paste0("joint_log_likelihood=", format(final_log_likelihood, digits = 15)),
    paste0("function_evaluations=", paste(fit$counts, collapse = ",")),
    paste0("message=", if (is.null(fit$message)) "none" else fit$message)
  ),
  file.path(output_dir, "optimizer_summary.txt")
)

estimates <- data.frame(
  parameter = names(fit$par),
  estimate = as.numeric(fit$par),
  stringsAsFactors = FALSE
)
write.csv(
  estimates,
  file.path(output_dir, "parameter_estimates.csv"),
  row.names = FALSE
)

per_tree_log_likelihoods <- tree_log_likelihoods(fit$par)
tree_summary <- data.frame(
  tree = names(tree_data),
  tips = vapply(tree_data, function(item) ape::Ntip(item$tree), integer(1)),
  stem_length_removed = vapply(tree_data, `[[`, numeric(1), "stem_length"),
  tips_A = vapply(tree_data, function(item) item$tip_counts[[1]], integer(1)),
  tips_B = vapply(tree_data, function(item) item$tip_counts[[2]], integer(1)),
  tips_C = vapply(tree_data, function(item) item$tip_counts[[3]], integer(1)),
  tips_D = vapply(tree_data, function(item) item$tip_counts[[4]], integer(1)),
  log_likelihood = as.numeric(per_tree_log_likelihoods),
  stringsAsFactors = FALSE
)
write.csv(
  tree_summary,
  file.path(output_dir, "tree_summary.csv"),
  row.names = FALSE
)

saveRDS(
  list(
    fit = fit,
    initial_values = starting_values,
    initial_log_likelihood = initial_log_likelihood,
    final_log_likelihood = final_log_likelihood,
    estimates = estimates,
    tree_summary = tree_summary,
    excluded_single_tip_trees = excluded_single_tip_trees,
    sampling_fraction = sampling_fraction,
    allowed_transitions = c("A->B", "A->C", "A->D"),
    root_handling = "Unary root stem dropped with ape::collapse.singles(root.edge=FALSE)",
    root_conditioning = "MuSSE ROOT.OBS",
    condition_survival = TRUE,
    trees = lapply(tree_data, function(item) item[c("tree", "states")])
  ),
  file.path(output_dir, "joint_musse_fit.rds")
)

summary_lines <- c(
  paste0("batch_dir=", batch_dir),
  paste0("trees=", length(tree_data)),
  paste0("excluded_single_tip_trees=", nrow(excluded_single_tip_trees)),
  paste0(
    "excluded_tree_ids=",
    if (nrow(excluded_single_tip_trees)) {
      paste(excluded_single_tip_trees$tree, collapse = ",")
    } else {
      "none"
    }
  ),
  paste0("sampling_fraction=", paste(sampling_fraction, collapse = ",")),
  "allowed_transitions=A->B,A->C,A->D",
  "other_transition_rates=0",
  "root_stem=dropped",
  "root_conditioning=ROOT.OBS",
  "condition_survival=TRUE",
  paste0("optimizer=L-BFGS-B; maxit=", maxit),
  paste0("convergence_code=", fit$convergence),
  paste0("joint_log_likelihood=", format(final_log_likelihood, digits = 15)),
  paste0("function_evaluations=", paste(fit$counts, collapse = ",")),
  paste0("message=", if (is.null(fit$message)) "none" else fit$message)
)
writeLines(summary_lines, file.path(output_dir, "joint_musse_fit.txt"))
cat("Convergence code:", fit$convergence, "\n")
cat("Final joint log likelihood:", final_log_likelihood, "\n")
cat("Results written to:", normalizePath(output_dir), "\n")
