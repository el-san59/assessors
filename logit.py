import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv('reformed_assessors_data.csv', index_col='id')
features, labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16], 11
data, target = df.values[:, features], df.values[:, labels]

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.4)

scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)
x_train = scaler.transform(x_train)

features = [11, 13, 0, 9]
if features:
    model = LogisticRegression()
    model.fit(x_train[:, features], y_train)
    print("Feature [{}]: {}".format(features, model.score(x_test[:, features], y_test)))
for i in [j for j in range(16) if j not in features]:
    model = LogisticRegression()
    model.fit(x_train[:, [*features, i]], y_train)
    print("Feature [{}]: {}".format(i, model.score(x_test[:, [*features, i]], y_test)))

