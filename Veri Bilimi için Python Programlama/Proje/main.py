import pandas as pd

# Görev 1: Aşağıdaki Soruları Yanıtlanıyınız.
# Soru 1.1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("Projeler/Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri/persona.csv")
df.head()

# Soru 1.2: Kaç unique "SOURCE" vardır? Frekansları nedir?
df["SOURCE"].unique()  # Kaç unique değer ve kaç adet?

# Soru 1.3: Kaç unique "PRICE" vardır?
df["PRICE"].unique()  # Kaç unique değer ve kaç adet?

# Soru 1.4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()  # Hangi PRICE'dan kaçar tane?

# Soru 1.5: Hangi ülkeden("COUNTRY") kaçar tane satış olmuş?
df["COUNTRY"].value_counts()  # Hangi COUNTRY'den kaçar tane?

# Soru 1.6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()  # COUNTRY'lere göre toplam PRICE.

# Soru 1.7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()  # SOURCE türüne göre satış kayıtları.

# Soru 1.8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean()  # COUNTRY'lere göre ortalama PRICE.

# Soru 1.9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()  # SOURCE'lara göre ortalama PRICE.

# Soru 1.10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()  # Ülke-Platform Bazlı ortalama fiyat.


# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()


# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
# Not: Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu
# azalan olacak şekilde PRICE’a göre uygulayınız.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
# Not: df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean() olarak kurgularsanız,
# çıktınız pandas.core.series.Series tipinde olacağından "sort_values" yapamazsınız.
agg_df = agg_df.sort_values("PRICE", ascending=False)


# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
# Not: Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler
# index isimleridir. Bu isimleri değişken isimlerine çeviriniz.
agg_df = agg_df.reset_index()


# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’eekleyiniz.
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_70'
def age_to_cat(x):
    if 0 <= x <= 18:
        return "0_18"
    elif 19 <= x <= 23:
        return "19_23"
    elif 24 <= x <= 30:
        return "24_30"
    elif 31 <= x <= 40:
        return "31_40"
    elif 41 <= x <= 70:
        return "41_70"
    else:
        return "70+"


agg_df["AGE_CAT"] = agg_df["AGE"].apply(lambda x: age_to_cat(x))


# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
# Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve
# veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
# değişkenini oluşturmanız gerekmektedir.
# Not: Dikkat! Listcomprehensionile customers_level_baseddeğerleri oluşturulduktan sonra bu değerlerin
# tekilleştirilmesi gerekmektedir. Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18.
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir
agg_df["customers_level_based"] = [country.upper() + "_" + source.upper() + "_" +
        sex.upper() + "_" + age_cat.upper() for country, source, sex, age_cat
        in zip(agg_df["COUNTRY"], agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT"])]

agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()


# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız
# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz (Segmentlere göre groupby yapıp price mean, max, sum’larını alınız).
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels=["D", "C", "B", "A"])


def analysis_segments(df):
    segment_list = list(df["SEGMENT"].unique())
    for segment in segment_list:
        print("Segment: ", segment,
              "\n", df[df["SEGMENT"] == segment].agg({"PRICE": ["mean", "max", "sum"]}), end="\n\n")


analysis_segments(agg_df)


# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.


def guess_income(user_segment): return list(agg_df[agg_df["customers_level_based"] == user_segment]["SEGMENT"]),\
                                       float(agg_df[agg_df["customers_level_based"] == user_segment]["PRICE"])


# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir?
user_1 = "TUR_ANDROID_FEMALE_31_40"
guess_income(user_1)

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
user_2 = "FRA_IOS_FEMALE_31_40"
guess_income(user_2)
