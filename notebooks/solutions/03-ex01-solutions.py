from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

hist = HistGradientBoostingRegressor(random_state=42)
hist.fit(X_train, y_train)
hist_pred = hist.predict(X_test)

compute_metrics(y_test, hist_pred)

hist_poisson = HistGradientBoostingRegressor(loss='poisson', random_state=42)
hist_poisson.fit(X_train, y_train)

hist_poisson_pred = hist_poisson.predict(X_test)

compute_metrics(y_test, hist_poisson_pred)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
ax1.hist(y_test, bins=30, alpha=0.5)
ax1.set_title("Test data")
ax2.hist(hist_pred, bins=30, alpha=0.5)
ax2.set_title("Default Hist")
ax3.hist(hist_poisson_pred, bins=30, alpha=0.5)
ax3.set_title("Poisson Hist");
