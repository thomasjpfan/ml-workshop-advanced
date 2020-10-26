base_rf.fit(X_train, y_train)
under_rf.fit(X_train, y_train)
over_rf.fit(X_train, y_train);

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
plot_roc_curve(base_rf, X_test, y_test, ax=ax1, name="original")
plot_roc_curve(under_rf, X_test, y_test, ax=ax1, name="undersampling")
plot_roc_curve(over_rf, X_test, y_test, ax=ax1, name="oversampling")

plot_precision_recall_curve(base_rf, X_test, y_test, ax=ax2, name="original")
plot_precision_recall_curve(under_rf, X_test, y_test, ax=ax2, name="undersampling")
plot_precision_recall_curve(over_rf, X_test, y_test, ax=ax2, name="oversampling");
