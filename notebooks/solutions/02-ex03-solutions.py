from sklearn.ensemble import HistGradientBoostingClassifier

base_hist = HistGradientBoostingClassifier(random_state=42)
base_hist.fit(X_train, y_train)

smote_hist = make_imb_pipeline(
    SMOTE(random_state=42), HistGradientBoostingClassifier(random_state=42))
smote_hist.fit(X_train, y_train)


plot_roc_and_precision_recall_curves(
    [
        ("original", base_hist),
        ("smote", smote_hist),
    ]
)