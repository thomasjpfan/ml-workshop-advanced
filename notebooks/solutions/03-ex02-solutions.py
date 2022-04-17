poisson_reg = Pipeline([
    ("preprocessor", linear_model_preprocessor),
    ("reg", PoissonRegressor(alpha=1e-4)),
])

poisson_reg.fit(X_train, y_train, reg__sample_weight=exposure_train)

poisson_pred = poisson_reg.predict(X_test)
compute_metrics(y_test, poisson_pred, sample_weight=exposure_test)

fig, ax = plt.subplots(figsize=(8, 8))
plot_calibration_curve_regression(y_test, poisson_pred, ax=ax, title="Poisson", sample_weight=exposure_test);
