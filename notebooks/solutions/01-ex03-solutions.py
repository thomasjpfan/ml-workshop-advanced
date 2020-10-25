from sklearn.datasets import fetch_20newsgroups
categories = [
    'alt.atheism',
    'sci.space',
]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

X_train, y_train = data_train.data, data_train.target

log_reg_tfid = Pipeline([
   ('vectorizer', TfidfVectorizer(stop_words="english")),
   ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

log_reg_tfid.fit(X_train, y_train)

X_test, y_test = data_test.data, data_test.target

log_reg_tfid.score(X_test, y_test)

feature_names = log_reg_tfid['vectorizer'].get_feature_names()
fig, ax = plt.subplots(figsize=(15, 6))
plot_important_features(log_reg_tfid['classifier'].coef_.ravel(), feature_names, top_n=20, ax=ax)
