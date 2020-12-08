from sklearn import linear_model
import pandas as pd
from scrape import scrape_data
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

dgci, x_training_data, x_test_data, y_training_data, y_test_data = scrape_data(1)

# Lasso regressor
print("Lasso shrinkage values vs r^2")
for i in range(1, 10):
    clf = linear_model.Lasso(alpha=i * 0.1)
    clf.fit(x_training_data, y_training_data)
    clf_y_hat = clf.predict(x_test_data)
    clf_r2 = r2_score(y_test_data, clf_y_hat)
    print(i, clf_r2)
