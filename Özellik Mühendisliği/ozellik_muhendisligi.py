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
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load_application_train():
    data = pd.read_csv('Özellik Mühendisliği/datasets/application_train.csv')
    return data


start_time = time.time()
df = load_application_train()
print(df)
print("--- %.4f seconds ---" % (time.time() - start_time))

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
                               and str(dataframe[col].dtypes) in ["uint8", "int32", "int64", "float64"]]
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
low, up = outlier_thresholds(df, "Fare")
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
scores.plot(stacked=True, xlim=[0, 700], style='.-')
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

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"Target Mean": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)


# Label Encoding & Binary Encoding
# Binary Encoding 0-1
# Label Encoding 0-1-2-..-n

df = load()
df.head()
df["Sex"].head()
le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)


df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

df = load()
df["Embarked"].value_counts()
df["Embarked"].nunique()
len(df["Embarked"].unique())

# One-Hot Encoding.
# Örneğin 4 değişken var.
# FB-GS-BJK-TS
# Sütunların şöyle olduğunu varsayalım.
# FB GS BJK TS
# 1  0   0  0
# 0  1   0  0
# 0  0   1  0
# 0  0   0  1
# Burada, FB sütunun olması, aslında bir dummy değişkendir.
# FB sütunu olmamalı,
# GS BJK TS
# 0   0  0
# 1   0  0
# 0   1  0
# 0   0  1
# Böylece hepsinin 0 olduğu yer FB.

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = load()

num_cols, cat_cols, cat_but_car = grab_col_names(df)

ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]

# Rare Encoding
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi
# 2. Rare kategoriler ile bağımlı değişken ilişksinin analiz edilmesi.
# 3. Rare encoder yazacağız.


# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

num_cols, cat_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, categorical_cols, plot=False):
    """
    Tasks
    -----
    Veri setindeki kategorik sütunlara ait verileri hakkında bilgiler veren fonksiyon.
    Parameters
    ----------
    dataframe: pandas.DataFrame
        İncelenmek istenilen pandas.Dataframe'i
    categorical_cols: list
        Kategorik sütunların olduğu liste
    plot: bool, default=False
        true ise, bir plot döndürür.
    Returns
    -------
    col:
        Sütun içerisinde bulunan benzersiz değerlerin listesi.
    Ratio:
        Bu değerlerin tüm veri setine dağılımı, değerler toplamı 1~ olmalı.
    """
    for col in categorical_cols:
        if (dataframe[col].dtypes == "category" or dataframe[col].dtypes == "bool") and dataframe[col].nunique() == 2:
            dataframe[col] = dataframe[col].astype(int)
        print("col type:", str(dataframe[col].dtype))
        print(pd.DataFrame({col: dataframe[col].value_counts(),
                            "Ratio": df[col].value_counts(normalize=True)}))
        print("#" * 23)
        if plot:
            sns.countplot(x=dataframe[col], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)


# Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, categorical_columns):
    for col in categorical_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)
rare_analyser(new_df, "TARGET", cat_cols)
df["OCCUPATION_TYPE"].value_counts()

# StandardScaler: Klasik Standartlaştırma. Ortalamayı Çıkar, standart sapmaya böl.
# z = (x - u) / s
# Standart sapma aykırı değerlerden etkilenebilir.

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])


# RobustScaler: Medyanı çıkar, IQR'a böl.
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df[["Age", "Age_standard_scaler", "Age_robuts_scaler"]].describe().T


# Min_max_scaler = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis  = 0))
# X_scaled = X_std * (max - min) * min

mns = MinMaxScaler()
df["Age_min_max_scaler"] = mns.fit_transform(df[["Age"]])
df[[col for col in df.columns if "Age" in col]].describe().T


def num_summary(dataframe, numerical_col, plot=False):
    """
    Tasks
    -----
    nümerik sütunlara ait describe() bilgilerini ve plot=True ise grafik bilgisini döndüren kod.
    Parameters
    ----------
    dataframe: pandas.DataFrame
        İncelenmek istenilen pandas.Dataframe'i
    numerical_col: list
        Nümerik sütunlara ait liste.
    plot: bool, default=False
        True olması halinde söz konusu sütunlara ait histogram'ın döndürülmesini sağlar.
    Returns
    -------
    Sütunlara ait describe bilgisini ve plot=True ise histogram grafiğini döndürür.
    """
    for col in numerical_col:
        print(col)
        print(dataframe[col].describe([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]).T)
        if plot:
            dataframe[col].hist()
            plt.xlabel(col)
            plt.title(col)
            plt.show(block=True)


num_cols = [col for col in df.columns if "Age" in col]

num_summary(df, num_cols, plot=True)

# Numeric to Categorical: Sayısal Değişkenleri Kategorik değişkenlere çevirme
# Binning

df["Age_qcut"] = pd.qcut(df["Age"], 5)


# Feature extraction (Özellik Çıkarımı)
df = load()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
df["NEW_CABIN_BOOL"].head()

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stats, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                       nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test stat= %.4f, p-value %.4f' % (test_stats, pvalue))
# pvalue, < 0.05 ise h0 hipotezi reddedilir, h1 kabul edilir.
# h0: Kabin numarası olması olmaması arasında survived bakımından anlamlı fark yoktur
# h1: Kabin numarası olması olmaması arasında survived bakımından anlamlı fark vardır

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stats, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                       nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test stat= %.4f, p-value %.4f' % (test_stats, pvalue))


# Text'ler üzerinden özellik türetmek.

df.head()

# Letter Count
df["NEW_NAME_COUNT"] = df["Name"].str.len()

# Word count
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

# Özel Yapıları Yakalamak
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

# Regex ile Değişken Türetmek

df = load()
df.head()

df["NEW_TITLE"] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=True)

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})


# Data Değişkenleri Üretmek

dff = pd.read_csv("Özellik Mühendisliği/datasets/course_reviews.csv")
dff.head()
dff.info()

dff["Timestamp"] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff["year"] = dff["Timestamp"].dt.year

# month
dff["month"] = dff["Timestamp"].dt.month

# year diff
dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year


dff['mont_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff["Timestamp"].dt.month

dff['day_name'] = dff['Timestamp'].dt.day_name()


# Feature Interactions (Özellik Etkileşimleri)

df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.groupby("NEW_SEX_CAT")["Survived"].mean()



# Feature Engineering
df = load()
df.columns = [col.upper() for col in df.columns]

# 1. Feature Engineering (Değişken Mühendisliği)
# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

num_cols, cat_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]


# Check outlier
check_outlier(df, num_cols)

# replace with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

# Check outlier
check_outlier(df, num_cols)

# Missing Values

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


# Label Encoding

binary_cols = [col for col in df.columns if df[col].dtype not in ["int32", "int64", "float64"] and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# Rare Encoding

rare_analyser(df, "SURVIVED", cat_cols)

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]

df = one_hot_encoder(df, ohe_cols)
df.head()
df.shape

num_cols, cat_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]


df.drop(useless_cols, axis=1, inplace=True)

# Standart Scaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

df.head()
y = df["SURVIVED"]
x = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Hiçbir işlem yapılmadan elde edilecek skor
dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train, save=True)

