import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#1 Первичный анализ: первые 5 строк, столбцы и их типы,и общий анализ
df = pd.read_csv('adult_train.csv', sep=';', header=0)

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

#2

# Визуализируем распределение концентраций по Hours_per_week - сколько человек отработал часов в неделю
sns.histplot(df['Hours_per_week'], kde=True)
plt.show()

# Визуализируем распределение по Age возраст
sns.histplot(df['Age'], kde=True)
plt.show()

# Визуализируем корреляцию между переменными
sns.heatmap(df.corr())
plt.show()
#4

# Удаляем текстовые поля так как они не нужны в анализе
df.drop(['Workclass','Education','Martial_Status','Occupation','Relationship','Race','Sex','Country','Target'], axis=1, inplace=True)




# Заменяем значения 0 на среднее значение/медиану

for i in df[df['Capital_Gain'].isnull()].index:
    df.loc[i, 'Capital_Gain'] = df['Capital_Gain'].mean()
for i in df[df['Capital_Loss'].isnull()].index:
    df.loc[i, 'Capital_Loss'] = df['Capital_Loss'].mean()


#5

# У этого набора данных нет класса.
# Поэтому используем кластеризацию k-средних для заполнения класса(прост и достаточно точен)
# Выбрано 2 кластера, потому что в описании данных указано,
# что датчик был расположен на поле в значительно загрязненной зоне.
# Таким образом, кластер 0 представляет собой ОЧЕНЬ сильно загрязненный,
# а кластер 1 представляет собой сильно загрязненный.

km = KMeans(n_clusters=2, random_state=1)
new = df._get_numeric_data()
km.fit(new)
predict = km.predict(new)
df['Class'] = pd.Series(predict, index=df.index)

# Разделяем набор данных на объекты и целевую переменную
X = df.drop('Class', axis=1)
y = df.loc[:, 'Class'].values

#6

# Выделяем данные на обучающие и тестовые наборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

param_grid = {'n_estimators': [10, 50, 100, 500],
              'max_depth': [5, 10, None]}

# Создай random forest regressor
rf = RandomForestRegressor(random_state=0)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

# Обучение
grid_search.fit(X_train, y_train)

#7

# Определяем диапазон размеров тестовой выборки
train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes, train_scores, validation_scores = learning_curve(
    rf, X_train, y_train, train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, validation_scores_mean, label='Validation error')
plt.xlabel('Number of training samples')
plt.ylabel('Mean squared error')
plt.title('Learning curve')
plt.legend()
plt.show()

# Определяем диапазон значений параметра
max_depth_range = range(1, 21)

train_scores, validation_scores = validation_curve(
    rf, X_train, y_train, param_name='max_depth', param_range=max_depth_range, cv=5, scoring='neg_mean_squared_error')

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(max_depth_range, train_scores_mean, label='Training error')
plt.plot(max_depth_range, validation_scores_mean, label='Validation error')
plt.xlabel('max_depth')
plt.ylabel('Mean squared error')
plt.title('Validation curve')
plt.legend()
plt.show()

#8

# Определяем конечную модель с max_depth=7
model = DecisionTreeRegressor(max_depth=7)

# Обучите модель на всей обучающей выберки
model.fit(X_train, y_train)

# делаем предсказания
y_pred = model.predict(X_test)

# Рассчитываем среднеквадратичную ошибку и R-квадрат оценки
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Среднеквадратичная ошибка', mse)
print('Оценка в R-квадрат', r2)
