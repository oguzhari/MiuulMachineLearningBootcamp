#########################################################
# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

##########################################################

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

##########################################################
# Adım 1: Key değerlerine erişiniz.

dict.keys()
# Çıktı: dict_keys(['Christian', 'Daisy', 'Antonio', 'Dante'])

##########################################################
# Adım 2: Value'lara erişiniz.

dict.values()
# Çıktı: dict_values([['America', 18], ['England', 12], ['Spain', 22], ['Italy', 25]])

##########################################################
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak günceleyiniz.

dict['Daisy'][1] = 13
# Çıktı: dict['Daisy'] = ['England', 13]

##########################################################
# Adım 4: Key değeri Ahmet, value değeri ["Turkey", 24] olan yeni bir değer ekleyiniz.

dict['Ahmet'] = ["Turkey", 24]
# Çıktı: 'Ahmet': ['Turkey', 24]

##########################################################
# Adım 6: Antonio'yu dictionary'den siliniz.

#Alternatif 1
dict.pop('Antonio')

#Alternatif 2
del dict['Antonio']
