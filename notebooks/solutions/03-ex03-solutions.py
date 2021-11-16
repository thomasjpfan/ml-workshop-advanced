from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector

tree_preprocessor = ColumnTransformer([
    ("categorical", 
     OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
         make_column_selector(dtype_include='object')
    ),
    ("numerical", "passthrough", make_column_selector(dtype_include='number'))
])

hist_poisson = Pipeline([
    ("preprocessor", tree_preprocessor),
    ("hist", HistGradientBoostingRegressor(loss="poisson", random_state=42))
])

hist_poisson.fit(X_train, y_train, hist__sample_weight=exposure_train)

hist_poisson_pred = hist_poisson.predict(X_test)

compute_metrics(y_test, hist_poisson_pred, sample_weight=exposure_test)

plot_calibration_curve_regression(y_test, hist_poisson_pred, sample_weight=exposure_test, title="Hist Poisson")
