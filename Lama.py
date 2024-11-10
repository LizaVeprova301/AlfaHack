import glob
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split

def getDataTrain():
    path_train = 'C:/Users/lizav/PycharmProjects/Lama4/train'
    # Получим список путей к файлам в папке train
    filenames_train = glob.glob(path_train + "/*.csv")

    # Создадим список для записи считанных файлов train
    data_files_train = []

    # Считаем все файлы train и добавим их в список
    for filename in filenames_train:
        data_files_train.append(pd.read_csv(filename))

    # Объединим тренировочные данные в единый датасет
    data_train = pd.concat(data_files_train, ignore_index=True)

    # Выведем информацию о размерности полученных тренировочных данных
    print('Размерность полных тренировочных данных составляет: {} строка и {} столбец'.format(*data_train.shape))

    # Выведем первые 5 строк тренировочных данных
    print(data_train.head())

    # Удостоверимся, что перед нами данные только из выборки train
    print(data_train['smpl'].value_counts(dropna=False))

    return data_train

def getDataTest():
    path_test = 'C:/Users/lizav/PycharmProjects/Lama4/test'
    # Получим список путей к файлам в папке test
    filenames_test = glob.glob(path_test + "/*.csv")

    # Создадим список для записи считанных файлов test
    data_files_test = []

    # Считаем все файлы test и добавим их в список
    for filename in filenames_test:
        data_files_test.append(pd.read_csv(filename))

    # Объединим тестовые данные в единый датасет
    data_test = pd.concat(data_files_test, ignore_index=True)

    # Выведем информацию о размерности полученных тестовых данных
    print('Размерность полных тестовых данных составляет: {} строк и {} столбцов'.format(*data_test.shape))

    # Выведем первые 5 строк тестовых данных
    print(data_test.head())

    # Удостоверимся, что перед нами данные только из выборки test
    print(data_test['smpl'].value_counts(dropna=False))
    return data_test


def validation(data_tr,data_te):
    data_train = data_tr.drop(columns=['smpl', 'id'])
    data_test = data_te.drop(columns=['smpl', 'id'])

    train_features = set(data_train.columns)
    test_features = set(data_test.columns)

    extra_train_features = train_features - test_features
    extra_test_features = test_features - train_features

    y_train = data_train['target']
    X_train = data_train.drop(columns=['target'])

    data_train = data_train.drop(columns=extra_train_features, errors='ignore')
    data_test = data_test.drop(columns=extra_test_features, errors='ignore')

    print("Финальные размеры:")
    print("Размер тренировочных данных:", data_train.shape)
    print("Размер тестовых данных:", data_test.shape)

    X_test = data_test

    print("Размер X_train:", X_train.shape)
    print("Размер y_train:", y_train.shape)
    print("Размер X_test:", X_test.shape)

    missing_train = X_train.isnull().sum()
    missing_test = X_test.isnull().sum()

    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for column in X_train.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        X_train[column] = label_encoders[column].fit_transform(X_train[column])
        X_test[column] = label_encoders[column].transform(X_test[column])

    from sklearn.preprocessing import StandardScaler

    numerical_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    print("Размер X_train после предобработки:", X_train.shape)
    print("Размер X_test после предобработки:", X_test.shape)

    import numpy as np
    correlation_threshold = 0.8
    corr_matrix = X_train.corr().abs()

    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]

    X_train_reduced = X_train.drop(columns=to_drop)
    X_test_reduced = X_test.drop(columns=to_drop)

    print("Исключенные признаки:", to_drop)
    print("Новые размеры данных после корреляционного отбора признаков:")
    print("Размер X_train_reduced:", X_train_reduced.shape)
    print("Размер X_test_reduced:", X_test_reduced.shape)

    # номер 2
    from sklearn.feature_selection import mutual_info_regression, SelectFromModel
    import numpy as np
    import pandas as pd

    mi_scores = mutual_info_regression(X_train, y_train)
    mi_scores = pd.Series(mi_scores, index=X_train.columns)

    threshold = mi_scores.median()

    selected_features = mi_scores[mi_scores >= threshold].index
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]

    print("Порог значимости:", threshold)
    print("Количество выбранных признаков:", len(selected_features))
    print("Выбранные признаки:", selected_features)
    print("Размер X_train после отбора признаков:", X_train_reduced.shape)
    print("Размер X_test после отбора признаков:", X_test_reduced.shape)
    return selected_features


def getRoles():
    roles = {
        'target': 'target',
        'drop': ['id', 'smpl']
    }
    return roles


def settingAutml():
    task = Task('multiclass')

    N_THREADS = 6
    N_FOLDS = 6
    RANDOM_STATE = 42
    TIMEOUT = 1200
    TARGET_NAME = 'target'

    roles = {
        'target': TARGET_NAME,
        'drop': ['id', 'smpl']
    }

    automl = TabularAutoML(
        task=task,
        timeout=TIMEOUT,
        cpu_limit=N_THREADS,
        reader_params={'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE}
    )
    return automl



data_train =getDataTrain()
data_test =getDataTest()
print("Данные загружены")

selectedFeatures = validation(data_train,data_test).tolist()

X = data_train.drop(['id', 'smpl'], axis=1)
y = data_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.3)
print(f'Размеры выборок:')
print(f'X_train: {X_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_train: {y_train.shape}')
print(f'y_test: {y_test.shape}')
y_test2=y_test.copy()

currAutml = settingAutml()
# Обучим модель на тренировочной части тренировочных данных
currAutml.fit_predict(X_train, roles=getRoles(),train_features=selectedFeatures, verbose=1)
# Получим предсказание с вероятностями для валидационной части тренировочного датасета
y_pred = currAutml.predict(X_test)
print(y_pred.data)
y_pred_series = pd.Series(y_pred.data[:, 1])
roc_auc = roc_auc_score(y_test, y_pred_series)
print(f'Метрика roc-auc на валидационных данных имеет значение: {roc_auc}')



# Обучим модель на тренировочной части тренировочных данных
currAutml.fit_predict(X, roles=getRoles(), train_features=selectedFeatures, verbose=1)
# Для предсказания используем тестовый датасет с исключенным признаком smpl
y_test_pred = currAutml.predict(data_test.drop(['smpl'], axis=1))
print(y_test_pred.data)
# Переведем предсказание в формат Series
y_test_pred_series = pd.Series(y_test_pred.data[:, 1])

# Добавим данные предсказания к датасету
data_test['target'] = y_test_pred_series

# Сохраним итоговые данные об id и предсказаниях в формате csv
data_test[['id', 'target']].to_csv('baseline_submission_case1.csv', index=False)

