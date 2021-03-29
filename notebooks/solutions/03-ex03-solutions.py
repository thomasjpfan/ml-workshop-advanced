from sklearn.preprocessing import OrdinalEncoder

tree_preprocessor = ColumnTransformer(
    [
        ("categorical", OrdinalEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
        ("numeric", "passthrough",
            ["VehAge", "DrivAge", "BonusMalus", "Density"]),
    ]
)

hist_poisson = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("reg", HistGradientBoostingRegressor(loss="poisson", random_state=0))
])
hist_poisson.fit(X_train, y_train, reg__sample_weight=exposure_train)

hist_poisson_pred = hist_poisson.predict(X_test)
compute_metrics(y_test, hist_poisson_pred, sample_weight=exposure_test)

fig, ax = plt.subplots(figsize=(8, 8))
plot_calibration_curve_weights(y_test, hist_poisson_pred, ax=ax, title="Hist Poisson", sample_weight=exposure_test);
