base_rf.fit(X_train, y_train)
under_rf.fit(X_train, y_train)
over_rf.fit(X_train, y_train)

plot_roc_and_precision_recall_curves([
    ("original", base_rf),
    ("undersampling", under_rf),
    ("oversampling", over_rf),
])