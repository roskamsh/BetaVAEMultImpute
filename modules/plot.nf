process PLOT_ACROSS_BETAS {
    publishDir "${params.outdir}/tune_beta", mode: "copy"
    label 'lasso'

    input:
    tuple val(betas), path(stats), path(logliks), val(coverage_levels_to_plot)

    output:
    path "MAE_EC_by_beta.pdf"
    path "MAE_EC_tradeoff_by_beta.pdf"
    path "Approx_loglikelihood_ymis_given_yobs_by_beta.pdf"

    script:
    def toRVector = { list, isNumeric = false, useBaseName = false ->
        if (isNumeric) {
            return "c(" + list.join(", ") + ")"
        } else if (useBaseName) {
            return "c(" + list.collect { 
                // Handle both Path objects and strings
                def pathStr = it.toString()
                def fileName = pathStr.split('/').last()
                def baseName = fileName.lastIndexOf('.') > 0 ? 
                    fileName.substring(0, fileName.lastIndexOf('.')) : fileName
                "\"${baseName}\""
            }.join(", ") + ")"
        } else {
            return "c(" + list.collect { "\"${it.toString()}\"" }.join(", ") + ")"
        }
    }
    """
    #!/usr/bin/env Rscript
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(stringr)
    library(cowplot)

    betas <- ${toRVector(betas, true)}
    stats_files <- ${toRVector(stats)}
    loglik_files <- ${toRVector(logliks)}
    coverage_levels_to_plot <- ${toRVector(coverage_levels_to_plot, true)}

    summarized_stats <- data.frame()
    summarized_logliks <- data.frame()
    for (i in 1:length(betas)) {
        stats_df <- read.csv(stats_files[i], header = F) 
        loglik_df <- read.csv(loglik_files[i])
        colnames(stats_df) <- c("variable","value")
        stats_df_t <- as.data.frame(t(stats_df\$value))
        colnames(stats_df_t) <- stats_df\$variable
        
        summarized_stats <- rbind(summarized_stats, stats_df_t)
        summarized_logliks <- rbind(summarized_logliks, loglik_df)
    }

    ## Now let's melt these dataframes so they are in the right format
    stats_forplot <- summarized_stats %>%
    pivot_longer(
        cols = starts_with("EC_") | starts_with("tradeoff_mae_ec_"),
        names_to = "metric",
        values_to = "value"
    ) %>%
    mutate(
        group = case_when(
        str_starts(metric, "EC_") ~ "EC",
        str_starts(metric, "tradeoff_mae_ec_") ~ "tradeoff_mae_ec"
        ),
        Coverage_level = str_extract(metric, "[0-9]+")
    ) %>%
    filter(as.numeric(Coverage_level) %in% coverage_levels_to_plot) %>%
    mutate(Coverage_level_label = paste("p=",Coverage_level,sep="")) %>%
    select(beta, imputation_strategy, MAE, group, Coverage_level, Coverage_level_label, value) %>%
    pivot_wider(
        names_from = group,
        values_from = value
    ) %>%
    mutate(MAE = as.numeric(MAE),
            EC = as.numeric(EC),
            tradeoff_mae_ec = as.numeric(tradeoff_mae_ec))

    mae_across_betas_plot <- stats_forplot %>%
    ggplot(mapping = aes(x = beta, y = MAE, group = Coverage_level_label)) +
    geom_point() +
    geom_line() +
    ylim(0,1) +
    theme_bw()

    hline_data <- stats_forplot %>%
    distinct(Coverage_level, Coverage_level_label) %>%
    mutate(Coverage_level = as.numeric(Coverage_level)/100)

    ec_across_betas_plot <- stats_forplot %>%
    ggplot(mapping = aes(x = beta, y = EC, colour = Coverage_level_label, group = Coverage_level)) +
    geom_point() +
    geom_line() +
    geom_hline(data = hline_data,
                aes(yintercept = Coverage_level, colour = Coverage_level_label),
                linetype = "dashed", alpha = 0.6, linewidth = 0.7) +
    scale_colour_brewer(palette = "Set1", name = "Coverage level") +
    ylim(0,1) +
    theme_bw()

    get_min <- stats_forplot %>%
    group_by(Coverage_level) %>%
    summarize(min_tradeoff = min(tradeoff_mae_ec)) %>%
    left_join(stats_forplot) %>%
    filter(min_tradeoff == tradeoff_mae_ec)
    
    tradeoff_across_betas_plot <- stats_forplot %>%
    ggplot(mapping = aes(x = beta, y = tradeoff_mae_ec, colour = Coverage_level, group = Coverage_level)) +
    geom_point() +
    geom_line() +
    geom_point(data = get_min, mapping = aes(x = beta, y = tradeoff_mae_ec), 
                shape = "asterisk", colour = "black", size = 8) +
    ylab("MAE + max(0,p-EC)") +
    scale_colour_brewer(palette = "Set1") +
    theme_bw()

    pdf("MAE_EC_by_beta.pdf", 10, 5)
    plot_grid(mae_across_betas_plot, ec_across_betas_plot)
    dev.off()

    pdf("MAE_EC_tradeoff_by_beta.pdf", 7, 5)
    print(tradeoff_across_betas_plot)
    dev.off()

    # need to write code to define beta hat and beta subopt
    beta_hat <- summarized_logliks %>%
    group_by(wrt) %>%
    summarize(
        beta_hat = beta[which.max(median_approxloglik)],
        max_loglik = max(median_approxloglik),
        .groups = "drop"
    )

    beta_subopt <- summarized_logliks %>%
    left_join(beta_hat, by = "wrt") %>%
    filter(
        median_approxloglik > bound_inf,
        beta >= beta_hat
    ) %>%
    group_by(wrt) %>%
    slice_min(beta, with_ties = FALSE) %>%
    ungroup() %>%
    select(wrt, beta, median_approxloglik) %>%
    mutate(type = "beta_subopt")

    beta_hat_plot <- beta_hat %>%
    transmute(wrt, beta = beta_hat, median_approxloglik = max_loglik, type = "beta_hat")
    highlight_points <- bind_rows(beta_hat_plot, beta_subopt)

    ## Approx loglikelihood plot
    approx_logilk_across_betas_plot <- ggplot(summarized_logliks, aes(x = beta, y = median_approxloglik)) +
    geom_point(size = 3) +
    geom_line() +
    geom_ribbon(aes(ymin = bound_inf, ymax = bound_sup), alpha = 0.2) +
    geom_point(
        data = filter(highlight_points, type == "beta_subopt"),
        aes(x = beta, y = median_approxloglik, shape = type, color = type),
        size = 6
    ) +
    geom_point(
        data = filter(highlight_points, type == "beta_hat"),
        aes(x = beta, y = median_approxloglik, shape = type, color = type),
        size = 6
    ) +
    scale_shape_manual(values = c("beta_hat" = 8, "beta_subopt" = 17)) +  # 8 = asterisk, 17 = triangle
    scale_color_manual(values = c("beta_hat" = "red", "beta_subopt" = "blue")) +
    facet_wrap(~ wrt) +
    labs(title = "Approximate loglikelihood Ymis given Yobs",
        y = "Approx. Log-Likelihood (median + IQ)",
        x = "Beta",
        shape = "Tuned betas",
        color = "Tuned betas") +
    theme_minimal()

    pdf("Approx_loglikelihood_ymis_given_yobs_by_beta.pdf", 7, 5)
    print(approx_logilk_across_betas_plot)
    dev.off()
    """
}