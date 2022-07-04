##########################################################
# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe
# çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime
# ayırınız.

# İpucu: String methodlarını kullanınız.
##########################################################

text = "The goal is to turn data into information, and information into sight"
text = text.replace("."," ")
text = text.replace(","," ")
text = text.upper().split(" ")
# Çıktı: ['THE',
#  'GOAL',
#  'IS',
#  'TO',
#  'TURN',
#  'DATA',
#  'INTO',
#  'INFORMATION',
#  '',
#  'AND',
#  'INFORMATION',
#  'INTO',
#  'SIGHT']


#@Hasan Alperen Albayrak'ın Önerdiği Alternatif
text.replace(".", " ").replace(",", " ").upper().split()
#https://stackoverflow.com/questions/3411771/best-way-to-replace-multiple-characters-in-a-string


#Ahmet Karayel'in Önerdiği Alternatif
char_to_replace = {",": " ",
                   ".": " "}

for key, value in char_to_replace.items():
  text = text.replace(key, value)
text
