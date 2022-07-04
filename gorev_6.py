##########################################################
# Görev 6: List Comprehension yapısı kullanarak car_crashes
# verisindeki numeric değişkenlerin isimlerini büyük harfe
# çeviriniz ve başına NUM ekleyiniz

# İpucu: Numeric olmayan değişkenlerin de isimleri büyümeli.
# İpucu: Tek bir list comprehension yapısı kullanılmalı.
##########################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

df.columns = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]
# Çıktı: Index(['NUM_TOTAL', 'NUM_SPEEDING', 'NUM_ALCOHOL', 'NUM_NOT_DISTRACTED',
#        'NUM_NO_PREVIOUS', 'NUM_INS_PREMIUM', 'NUM_INS_LOSSES', 'ABBREV'],
#        dtype='object')
