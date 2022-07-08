##########################################################
# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki
# tek ve çift sayıları ayrı listelere atayan ve bu listeleri
# return eden fonksiyonu yazınız.

# İpucu: Liste elemanlarına tek tek erişmeniz gerekmektedir.
# İpucu: Her bir eleman çift veya tek olma durumunu kontrol etmek için % yapısını kullanabilirsiniz.
##########################################################

l = [2, 13, 18, 93, 22]


def func(liste):
    even_list = []
    odd_list = []
    [even_list.append(i) if i % 2 == 0 else odd_list.append(i) for i in liste]
    return even_list, odd_list


even_list, odd_list = func(l)
# Çıktı: even_list = [2, 4], odd_list = [1, 3]
