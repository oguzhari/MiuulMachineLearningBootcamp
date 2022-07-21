import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


def load():
    """
    Tasks
    -----
    İhtiyaç duyulması halinde çağrılarak orijinal veri setini yeniden yükleme fonksiyonu.

    Returns
    -------
    diabetes.csv dosyasını pandas.dataFrame olarak döndürür.
    """
    dataframe = pd.read_csv("Özellik Mühendisliği/bitirme_projesi/diabetes.csv")
    return dataframe


df = load()
df.columns
# Pregnancies: Geçirilen hamilelik sayısı
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu (tokluk şekeri)
# BloodPressure: Kan basıncı, küçük tansiyon. (mm Hg)
# SkinThickness: Cilt kalınlığı
# Insulin: 2 saatlik serum insülini (mu U/ml) (Pankreasın ürettiği serbest insülin)
# DiabetesPedigreeFunction: Soydaki kişilere göre diyabet olma ihtimalini hesaplayan fonksiyon
# BMI: Vücut kitle endeksi
# Age: Yaş (yıl bazında)
# Outcome: 1-Diyabet, 0-Diyabet değil.

######################################
# Görev 1: Keşifçi Veri Analizi
######################################

####################################
# Adım 1.1: Genel resmi inceleyiniz.

def check_df(dataframe, head=5):
    print("##### Shape #####")
    print(dataframe.shape)
    print("\n########### Types ###########")
    print(dataframe.dtypes)
    print("\n################################ Head ################################")
    print(dataframe.head(head))
    print("\n################################ Tail ################################")
    print(dataframe.tail(head))
    print("\n######### NA #########")
    print(dataframe.isnull().sum())
    print("\n################################ Quantiles ################################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

##########################################################
# Adım 1.2: Numerik ve kategorik değişkenleri yakalayınız.


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

    numeric_cols = [col for col in dataframe.columns if dataframe[col].dtype in ["int64", "float64"]]
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]

    print(f"Observations -> {dataframe.shape[0]}")
    print(f"Variable     -> {dataframe.shape[1]}")
    print(f"cat_cols     -> {len(categorical_cols)}")
    print(f"num_cols     -> {len(numeric_cols)}")
    print(f"cat_but_car  -> {len(categorical_but_cardinal)}")
    print(f"num_but_cat  -> {len(numeric_but_categorical)}")

    return numeric_cols, categorical_cols, categorical_but_cardinal


num_cols, cat_cols, cat_but_car = grab_col_names(df, cat_th=20)
# Sayısal verilerin çokluğu sebebiyle th_20 olarak verildi.


##########################################################
# Adım 1.3: Numerik ve kategorik değişkenleri yakalayınız.


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts(normalize=True)}))
    print("####################")
    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    print(dataframe[numerical_col].describe([0, 0.05, 0.50, 0.95, 0.99]).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col)

for col in num_cols:
    num_summary(df, col)

##########################################################
# Adım 1.4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması).


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean(),
                        "Value_Count": dataframe.groupby(categorical_col)[target].count()}))


for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)
# Tek kategorik olarak yakalanan değişken Outcome.


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ["mean"]}), end="\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


##########################################################
# Adım 1.5: Aykırı gözlem analizi yapınız.


def outlier_thresholds(dataframe, col_name, q1_th=0.25, q3_th=0.75):
    quartile1 = dataframe[col_name].quantile(q1_th)
    quartile3 = dataframe[col_name].quantile(q3_th)
    inter_quantile_range = quartile3 - quartile1
    upper_limit = quartile3 + (1.5 * inter_quantile_range)
    lower_limit = quartile1 - (1.5 * inter_quantile_range)
    return lower_limit, upper_limit


def check_outlier(dataframe, col_names):
    print("_" * 36)
    print("{:25s} {}".format("Column", "True/False"))
    print("-" * 36)
    for col_name in col_names:
        lower_limit, upper_limit = outlier_thresholds(dataframe, col_name)
        print("{:25s} {}".format(col_name, dataframe[(dataframe[col_name] < lower_limit) |
                                                    (dataframe[col_name] > upper_limit)].any(axis=None)))


check_outlier(df, df.columns)


##########################################################
# Adım 1.6: Eksik gözlem analizi yapınız.


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

##########################################################
# Adım 1.7: Korelasyon analizi yapınız.


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (10, 10)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_cor_cols = high_correlated_cols(df)


######################################
# Görev 2: Feature Engineering
######################################

##########################################################
# Adım 2.1: Eksik değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin
# vb. değişkenlerde 0 değer içeren gözlem birimleri eksik
# değeri ifade ediyor olabilir. Örneğin, bir kişini glikoz
# veya insülin değeri 0 olamayacaktır. Bu durumu dikkate
# alarak sıfır değerlerini ilgili değerlerde NaN atama yapıp
# sonrasında değerlere işlemleri uygulayabilirsiniz.

check_df(df)
# Burada dataframe içerisinde min değerleri inceleyeceğiz.
# 0 olmasının anlamlı olduğu sütunları
# -> Pregnancies, insanlar daha önce hamilelik yaşamamış olabilir.
# -> Outcome, 0 olması diyabet olmaması anlamına gelmektedir.
# 0 olmasının anlamsız olduğu sütunlar;
# Glucose, BloodPressure, SkinThickness, Insulin, BMI

# Bu sütunlar üzerindeki 0 sayılarına bakalım
meaningless_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def check_zeros(dataframe, zero_cols):
    n_zero = dataframe[zero_cols].isin([0]).sum().sort_values(ascending=False)
    ratio = (dataframe[zero_cols].isin([0]).sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_zero, np.round(ratio, 2)], axis=1, keys=['n_zero', 'ratio'])
    print(missing_df, end="\n")


check_zeros(df, meaningless_zero)

# Sıfır gördüğümüz yerlere numpy.nan atamamız gerekiyor.

df = load()
# Hatalı bir işlem olması durumunda sıfırlamak adına.


def fill_n_nan(dataframe, col_names, rep=None):
    """
    Tasks
    -----
    Verilen dataframe'in istenilen sütunlara ait rep ile ifade edilmiş değerleri np.nan ile dolduran fonksiyon.

    Parameters
    ----------
    dataframe: pandas.DataFrame
        Hedef dataframe
    col_names: list
        Değişilmesi istenen sütunlara ait isimleri içeren list
    rep: NoneType
        Sütunların içerisinden değiştirilmesi istenilen değer. int, float ve string gibi farklı değerler kabul
        edebilir.

    Returns
    -------
    verilen sütunlarının verilen rep değeri ile değiştirilmiş dataframe.
    """
    for col in col_names:
        dataframe[col] = dataframe[col].replace(rep, np.nan)
    return dataframe[col_names]


df[meaningless_zero] = fill_n_nan(df, meaningless_zero, 0)

check_zeros(df, meaningless_zero)
missing_cols = missing_values_table(df)

# Atama işlemlerinden sonra, nan değerlerin bir makine
# öğrenimi yöntemiyle doldurulması gerektiğini düşünüyorum.
# Diyabet kişinin sağlılıyla direkt olarak ilgili olduğu için
# boş değerlerin sınırlara çekilesi değil, diğer değerlere göre
# doldurulması daha sağlıklı olacaktır.


Imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(Imputer.fit_transform(df), columns=df.columns)
# Boş olan değerleri KNNImputer ile doldurarak, elde ettiğimiz dataframe'i
# tekrardan üstünde çalıştığımız dataframe'e yazdık. Şu an dataframe'imizde
# nan ya da 0 olan bir değer yok.

check_zeros(df, meaningless_zero)
missing_values_table(df)

# Sıfırlardan kurtulduktan sonra yapmamız gereken şey, outliers'lardan kurtulmak
# Ancak direkt outliers'ı seçmek yerine, diğer değerlere göre outliers kontrolü yapmalıyız.
# Çünkü şeker hastalığı kişinin sağlık durumuyla doğrudan alakalı.
# outliers değerleri kendi özellerinde incelemeliyiz.

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5]

# describe ederek genele bakalım.
df_scores_des = pd.DataFrame(np.sort(df_scores))
df_scores_des.describe().T

# Görsel olarak kırılımı görelim.
scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()
plt.savefig('diabetes_outliers', dpi=300)

# Treshold değerimizi tanımlayalım.
th = np.sort(df_scores)[5]

# Veriye göz atalım.
df[df_scores < th]
df.describe([0.01, 0.05, 0.75, 0.9, 0.99]).T

# Aykırılardan kurtulalım.
df.drop(df[df_scores < th].index, axis=0, inplace=True)

check_df(df)


