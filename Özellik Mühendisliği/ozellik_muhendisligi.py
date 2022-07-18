# Özellik Mühendisliği.
# Aykırı değerler;
# Özellikle doğrusal problemlerde, aykırı değerlerin etkisi daha şiddetlidir.
# Aykırı değerler üzerinde işlemler için en önemli konulardan birisi eşik değerin belirlenmesi.
# Hangi noktadan sonrası aykırı, hangi noktanın altı normal değer olarak kabul edilecek.
#
# Interquartile Range (IQR)
# ||¹ - |² -------- [³  |⁴ ]⁵ -------- |⁶ - ||⁷
# Q3 - Q1 = IQR değeri.
# 1 - Lower 1.5 x IQR Whisker -> Q1 - 1.5 x IQR
# 2 - Minimum değer.
# 3 - Q1, %25. değer.
# 4 - Medyan, ortanca değer.
# 5 - Q3, %75. değer.
# 6 - Maximum değer.
# 7 - Upper 1.5 x IQR Whisker -> Q3 + 1.5 x IQR

# Feature Engineering & Data Pre-Processing

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv('Özellik Mühendisliği/datasets/application_train.csv')
    return data


df = load_application_train()
df.head()


def load():
    data = pd.read_csv('Özellik Mühendisliği/datasets/titanic.csv')
    return data


df = load()
df.head()

# Grafik Teknikle Aykırı Değerler
sns.boxplot(x=df["Age"])
plt.show(block=True)

# Aykırı Değerler Nasıl Yakalanır?

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
upper = q3 + (1.5 * iqr)
lower = q1 - (1.5 * iqr)
df[(df["Age"] < lower) | (df["Age"] > upper)]  # Aykırı değerler
df[(df["Age"] < lower) | (df["Age"] > upper)].index  # Aykırı değerlere ait indexler

# Aykırı değer var mı yok mu?
df[(df["Age"] < lower) | (df["Age"] > upper)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok mu diye sorduk


def outlier_thresholds(dataframe, col_name, q1_th=0.25, q3_th=0.75):
    quartile1 = dataframe[col_name].quantile(q1_th)
    quartile3 = dataframe[col_name].quantile(q3_th)
    inter_quantile_range = quartile3 - quartile1
    upper_limit = quartile3 + (1.5 * inter_quantile_range)
    lower_limit = quartile1 - (1.5 * inter_quantile_range)
    return lower_limit, upper_limit


outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

df[(df["Fare"] < lower) | (df["Fare"] > upper)].any(axis=None)


def check_outlier(dataframe, col_names):
    for col_name in col_names:
        lower_limit, upper_limit = outlier_thresholds(dataframe, col_name)
        print("{}: {}".format(col_name, dataframe[(dataframe[col_name] < lower_limit) |
                                                  (dataframe[col_name] > upper_limit)].any(axis=None)))


check_outlier(df, "Fare")
dff = load_application_train()


def grab_col_names(dataframe, cat_th=10, car_th=20) -> [list, list, list]:
    """
    Tasks
    ----------
    Analiz edilmek istenen dataframe için bir analiz fonksiyonu. Bu fonksiyon, almış olduğu dataframe'in numerik,
    kategorik ve kardinal sütunlarını ayrı ayrı olacak şekilde döndürecektir.
    Parameters
    ----------
    dataframe: pandas.Dataframe
        İncelenmek istenilen pandas.Dataframe'i
    cat_th: int, float default=20
        numeric olan sütunlar için categoric varsayılma eşiği. bu eşiğin altında bulunan int64 ve float64
        veri tipindeki sütunlar numeric değil, categoric olarak işlenecek.
    car_th: int, float default=10
        categoric olan sütunlar için cardinal varsayılma eşiği. bu eşiğin altında bulunan category, object ve bool
        veri tipindeki sütunlar categoric değil, cardinal olarak işlenecek.
    Returns
    -------
    numeric_cols: list
        numeric_cols, veri tipi "int64" ve "float64" veri tiplerine sahip ancak categorical_cols içinde bulunmayan
        sütunların listesi.
    categorical_cols: list
        categorical_cols, veri tipi "category", "object" ve "bool" olan,ancak kardinal olmayan ve
        veri tipi int64 ve float64 olup cat_th'dan daha az unique veri tipine sahip sütunların listesi.
    categorical_but_cardinal: list
        categorical_but_cardinal, veri tipi "category", "object" ve "bool" olancar_th değerinden daha fazla unique
        kategori içeren sütunların listesi.
    Observations: int
        Toplam satır sayısı.
    Variable: int
        Toplam sütun sayısı.
    cat_cols: int
        kategorik sütunların sayısı.
    num_cols: int
        numerik sütunların sayısı.
    cat_but_car: int
        kategorik ama kardinal olan sütunların sayısı.
    num_but_cat: int
        numerik ama kategorik olan sütunların sayısı.
    """
    categorical_cols = [col for col in dataframe.columns if str(dataframe[col].dtype) in ["category", "object", "bool"]]
    numeric_but_categorical = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                               and str(dataframe[col].dtypes) in ["int64", "float64"]]
    categorical_but_cardinal = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                                and str(dataframe[col].dtypes) in ["category", "object"]]
    categorical_cols = categorical_cols + numeric_but_categorical
    categorical_cols = [col for col in categorical_cols if col not in categorical_but_cardinal]

    numeric_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]

    print(f"Observations -> {dataframe.shape[0]}")
    print(f"Variable     -> {dataframe.shape[1]}")
    print(f"cat_cols     -> {len(categorical_cols)}")
    print(f"num_cols     -> {len(numeric_cols)}")
    print(f"cat_but_car  -> {len(categorical_but_cardinal)}")
    print(f"num_but_cat  -> {len(numeric_but_categorical)}")

    return numeric_cols, categorical_cols, categorical_but_cardinal


num_cols, cat_cols, cat_but_car = grab_col_names(df)


check_outlier(df, num_cols)


# Aykırı değerlerin kendilerine erişim
def grab_outliers(dataframe, col_name, index=False):
    lower_limit, upper_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))].head())
    else:
        print(dataframe[((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < lower_limit) | (dataframe[col_name] > upper_limit))].index
        return outlier_index


age_index = grab_outliers(df, "Age", True)

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

# Aykırı Değerleri Dövmek

# Silme
low, up = outlier_thresholds(df, "Age")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    lower_limit, upper_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe["Fare"] < lower_limit) | (dataframe["Fare"] > upper_limit))]
    return df_without_outliers


num_cols, cat_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)


df.shape[0] - new_df.shape[0]

# Baskılama yöntemi (re-assigment with thresholds)
low, up = outlier_thresholds(df, "Fare")

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] < up), "Fare"] = low


def replace_with_thresholds(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit


df = load()
num_cols, cat_cols, cat_but_car = grab_col_names(df)
df.shape

check_outlier(df, num_cols)


for col in num_cols:
    replace_with_thresholds(df, col)


check_outlier(df, num_cols)
grab_outliers(df, "Age", index=True)

remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")

# Çok değişkenli Aykırı Değer Analizi: Local Outlier Factor
# 3 kez evlenmiş olmak, bir aykırı değer midir? Hayır.
# 17 yaşında olmak bir aykırı değer midir? Hayır.
# Ancak, 17 yaşında olup 3 kez evlenmiş olmak bir aykırı değerdir.
# Bazı değerler birlikte alındığında aykırı değer olarak ifade edilebilir.
#
# LOFi -> Gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak, aykırı değer olabilecek değerleri
# tanıma imkanı sağlar.
# Bir noktanın lokal yoğunluğu, o noktanın etrafındaki komşuluklar demektir. O noktanın yoğunluğu komşularınkinden
# anlamlı bir şekilde düşükse, o nokta aykırı değerdir. LOFi bize aykırı değerleri hesaplama imkanı sağlar.

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

check_outlier(df, list(df.columns))

low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
# Negatif skorların olmasını istemiyorsak
# df_scores = -df_scores
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
th = np.sort(df_scores)[3]

df[df_scores < th].shape
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Ağaç yöntemleri kullanıyorsak, aykırı değerlere hiç ellememeyi tercih ediyoruz.
# Eğer dokunacaksak, sınırları %1-%99 veya %5-%95 seçebiliriz.
# Doğrusal yöntemlerde ise aykırı değerler çok önemlidir. Eğer aykırı değerler azsa silinebilir.
# Doldurmak yerine de, tek değişkenli yaklaşıp (değişken bazında) baskılamak kullanılabilir.

# Missing Values
# Gözlemlerde eksiklik olması durumunu ifade etmektedir.
# Eksik değerler nasıl çözüür?
# -> Silme
# -> Değer atama
# -> Tahmine dayalı yöntemler

# Eksik değerlerin yakalanması

df = load()
df.head()

# Eksik gözlem var mı yok mu sorgusu
df.isnull().values.any()

# değişkenkerdeki eksik değer sayısı.
df.isnull().sum()

# Değişkenlerdeki tam değer sayısı.
df.notnull().sum()

# Veri setindeki toplam eksik değer
df.isnull().sum().sum()

# en az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


# Eksik Değer Problemini Çözme
missing_values_table(df)

# Çözüm 1: Hızlıca Silmek
df.dropna()

# Çözüm 2: Basit Atama Yöntemleri ile doldurmak
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())


dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing")

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


# Kategorik Değişken Kırılımında Değer Atama

df.groupby("Sex")["Age"].mean()

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))
# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


# Çözüm 3: Tahmine Dayalı Atama ile Doldurma

df = load()

num_cols, cat_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

# Değişkenkerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# knn'in uygulanması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]


# Gelişmiş Analizler
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()


# Eksik değerlerin bağımlı değişken ile ilişkisinin incelenmesi



