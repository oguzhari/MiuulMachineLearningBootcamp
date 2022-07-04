##########################################################
# Görev 7: List Comprehension yapısı kullanarak car_crashes
# verisindeki isminde "no" BARINDIRMAYAN değişkenlerin sonuna
# "FLAG" yazınız.

# İpucu: Tüm değişkenlerin isimleri büyük harf olmalı.
# İpucu: Tek bir list comprehension yapısı kullanılmalı.
##########################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]
# Çıktı: Index(['TOTAL_FLAG', 'SPEEDING_FLAG', 'ALCOHOL_FLAG', 'NOT_DISTRACTED',
#        'NO_PREVIOUS', 'INS_PREMIUM_FLAG', 'INS_LOSSES_FLAG', 'ABBREV_FLAG'],
#        dtype='object')
