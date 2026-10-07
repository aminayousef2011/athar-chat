"""Intent recognizers: the only component that differs between systems.

S1 keyword rules (:mod:`athar.intent.rules`), S2 word TF-IDF + logistic regression and S3 S2 +
character n-grams (:mod:`athar.intent.models`). Frozen systems are loaded with
:func:`athar.intent.train.load_recognizer`.
"""
