##########################################################
# Görev 9: cat_summary() fonksiyonuna1 özellik ekleyiniz. Bu özellik argümanla biçimlendirilebilir olsun.
# Var olan özelliği de argümanla kontrol edilebilir hale getirebilirsiniz.

##########################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("titanic")


def cat_summary(dataframe, col_name, plot=False, pie=False):
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


cat_summary(df, "class", pie=True)

