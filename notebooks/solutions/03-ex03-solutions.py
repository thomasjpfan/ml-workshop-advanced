poisson_results = permutation_importance(poission_reg, X_test, y_test, scoring=neg_mean_poisson_deviance,
                                         n_repeats=10)

plot_permutation_importance(poisson_results, feature_names);
