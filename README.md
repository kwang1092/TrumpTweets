# TrumpTweets
created a model to determine if a tweet made by Trump was sent from an iPhone or an Android

Made multiple attempts using different features/classifiers for max accuracy:
- Attempted to use Facebook's InferSent sentence embeddings for features
- Used a combination of the various metadata in the training/test csvs
- Tried Logistic Regression, SVM, Naive Bayes, and Random Forests

Final Model:
- Decided to use Random Forests, as it minimized our validation/training error and tends to be one of the better models for lowering variance
- Ended up scrapping the sentence embeddings because accuracy wasn't improving (may try different preprocessing techniques/different embeddings in the future)

Results:
- Ended with ~80% test error
