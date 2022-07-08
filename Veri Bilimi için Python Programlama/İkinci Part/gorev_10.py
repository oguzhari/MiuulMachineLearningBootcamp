##########################################################
# Görev 10: check_df(), cat_summary() fonksiyonlarına 4 bilgi(uygunsa) barındıran numpy tarzı docstring yazınız.
# (task, params, return, example)

##########################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")


def check_df(dataframe, head=5):
    """
    Task
    ----------
    Dataframe hakkında genel bilgi alma amaçlı fonksiyon.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        Hakkında genel bilgi alınmak istenilen dataFrame.
    head: int
        Ekrana yazdırılacak head() ve tail() methodları için limit. default = 5

    Returns
    -------
    Shape
        dataframe.shape
        dataframe'e ait (satır, sütun) bilgileri.
    Types
        dataframe.dtypes
        dataframe'e ait sütun tip bilgileri.
    Head
        dataframe.head(head)
        dataframe'in ilk (head) satırı.
    Tail
        dataframe.tail(head)
        dataframe'in son (head) satırı.
    Na
        dataframe.isnull().sum()
        dataframe'e ait sütunlardaki boş değer sayısı.
    Quantiles
        dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T
        dataframe'e ait numerik değerlerin %0, %5, %50, %95, %99 ve %100. değerlerini.

    Notes
    -------
    Birden fazla dataframe analizi için for döngüsü kullanılabilir.

    for dataframe in dataframes:
        check_df(dataframe)

    """
    print("######## Shape ########")
    print(dataframe.shape)
    print("######## Types ########")
    print(dataframe.dtypes)
    print("######## Head ########")
    print(dataframe.head(head))
    print("######## Tail ########")
    print(dataframe.tail(head))
    print("########  Na  ########")
    print(dataframe.isnull().sum())
    print("####  Quantiles  #####")
    print(dataframe.describe([0, 0.05, 0.5, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False, pie=False):
    """
    Task
    ----
    Verilen dataframe'in belirtilen sütununa ait kategorik değişkenlere yönelik oransal bilgi veren,
    pie-chart ve histogram üzerinde görselleştirme sunan method.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        Analiz edilecek dataframe.
    col_name: str, int
        Analiz edilmek istenen sütunu adı ya da indeksi.
    plot: bool
        bar şeklinde histogram görselleştirmesi döndürülsün mü?
    pie: bool
        pie şeklinde kategorik dağılım gdöndürülsün mü?

    Returns
    -------
            sütun_adı      Ratio
    bir     int            float
    iki     int            float
    uc      int            float

    pie:
        kategori isimlerine sahip piechart.
    plot:
        kategorilere göre histogram.

    Notes
    -----
    true-false şeklinde kategoriler öncelikle astype(int) olarak dönüştürülmeli.
    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts(normalize=True)}))
    print("####################")
    if pie:
        plt.pie(dataframe[col_name].value_counts(), labels=list(dataframe[col_name].unique()))
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)
    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)





