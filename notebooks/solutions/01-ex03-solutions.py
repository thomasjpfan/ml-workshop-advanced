log_tfid = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words='english')),
    ("log_reg", LogisticRegression(solver='liblinear'))
])

len(text_train)

len(text_test)

log_tfid.fit(text_train, y_train)

log_tfid.score(text_test, y_test)

feature_names = log_tfid["vectorizer"].get_feature_names_out()
log_reg_coefs = log_tfid["log_reg"].coef_.ravel()

plot_important_features(log_reg_coefs, feature_names)
