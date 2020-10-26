from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

base_hist = HistGradientBoostingClassifier(random_state=42)
base_hist.fit(X_train, y_train)

smote_hist = make_imb_pipeline(
    SMOTE(), HistGradientBoostingClassifier(random_state=42))
smote_hist.fit(X_train, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
plot_roc_curve(base_hist, X_test, y_test, ax=ax1, name="original")
plot_roc_curve(smote_hist, X_test, y_test, ax=ax1, name="smote")

plot_precision_recall_curve(base_hist, X_test, y_test, ax=ax2, name="original")
plot_precision_recall_curve(smote_hist, X_test, y_test, ax=ax2, name="smote")
