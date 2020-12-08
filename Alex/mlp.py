from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from scrape import scrape_data
from sklearn.model_selection import cross_val_score

dgci, x_training_data, x_test_data, y_training_data, y_test_data = scrape_data(1)

# Neural Net
mlp = MLPRegressor(random_state=1, hidden_layer_sizes=100)
mlp.fit(x_training_data, y_training_data)
mlp_y_hat = mlp.predict(x_test_data)
mlp_r2 = r2_score(y_test_data, mlp_y_hat)
print(mlp_r2)
