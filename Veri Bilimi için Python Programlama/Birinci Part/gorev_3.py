##########################################################
# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.

##########################################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

##########################################################
# Adım 1: Verilen listenin eleman sayısına bakınız.

len(lst)
# Çıktı: 11

##########################################################
# Adım 2: Sıfırıncı ve onuncu indesteki elemanları çağırınız.

print(lst[0], lst[10])
# Çıktı: D E

##########################################################
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz

new_list = lst[0:4]
# Çıktı: new_list = ['D', 'A', 'T', 'A']

##########################################################
# Adım 4: Sekizinci indeksteki elemanı siliniz.

lst.remove(lst[8])
# Çıktı: ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E']

##########################################################
# Adım 5: Yeni bir eleman ekleyiniz.

lst.append("M")
# Çıktı: ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'C', 'E', 'M']

##########################################################
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

lst.insert(8, "N")
# Çıktı: ['D', 'A', 'T', 'A', 'S', 'C', 'I', 'E', 'N', 'C', 'E', 'M']
