##########################################################
# Görev 8: List Comprehension yapısı kullanarak aşağıda verilen
# değişken isimlerinden FARKLI olan değişkenlerin

# İpucu: Önce verilen listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# İpucu: Sonra df[new_cols] ile bu değişkenleri seçerek yeni birdf oluşturunuz ve adını new_df olarak isimlendiriniz.
##########################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
# Çıktı: ['total', 'speeding', 'alcohol', 'not_distracted', 'ins_premium', 'ins_losses']

new_df = df[new_cols]
new_df.head()
