import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT','MEDV']
data = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
df = pd.DataFrame(data)

df.to_csv('housing_df.csv', index=False)

df = pd.read_csv('housing_df.csv')

y = df['MEDV']
x = df.drop(['MEDV'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

parameters = {
    'lin_reg__fit_intercept': [True, False],
    'lin_reg__copy_X': [True, False],
    'lin_reg__positive': [True, False]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)

y_predictions = grid_search.predict(X_test)
y_predictions

mae = mean_absolute_error(y_test, y_predictions)
r2 = r2_score(y_test, y_predictions)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

filename = 'logistic_regression_model_V0.2.pkl'
joblib.dump(grid_search, filename)