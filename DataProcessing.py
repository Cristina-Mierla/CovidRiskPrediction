import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import csv

sns.set()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DataProcessing:

    def __init__(self, dataset):
        self.dataset = dataset
        self.df = pd.DataFrame(self.dataset)

        self.copy_dataset = self.dataset

        self.boala = None

        self.replace_columns()
        self.change_medicatie()
        self.analize_columns()
        self.comorbiditati = self.comorbiditati_columns()
        self.diagnos_columns()

        self.describe()

        self.comorb = self.df["Comorbiditati"]
        self.exetern = self.df["stare_externare"]
        self.varsta = self.df["Varsta"]
        self.spitalizare = self.df["Zile_spitalizare"]
        self.diagnos_int = self.df["Diag_pr_int"]
        self.com_ext = self.df[["Comorbiditati", "stare_externare"]]
        # ["Sex", "Varsta", "Zile_spitalizare", "zile_ATI", "Diag_pr_int", 'Analize_prim_set', "Comorbiditati", "AN", "precod", "FO",
        #  "Data_Examinare_Radiologie", "stare_externare", "forma_boala",  "Radiologie", "rezultat_radiologie", "Proceduri",
        #  "Proceduri_Radiologie", "tip_externare", "unde_pleaca", "Diag_pr_ext"]
        # self.data = [[self.__preg, self.__gluc, self.__blpr, self.__skth, self.__insu, self.__bmi, self.__pedi, self.__age]]

    def predict(self, prediction_data):
        # prediction_data = [age, sex, diagnos_int, spitalizare, ati, analize]

        age = prediction_data[0]
        sex = prediction_data[1]
        diag_init = prediction_data[2]
        zile_spit = prediction_data[3]
        zile_ati = prediction_data[4]
        analize = prediction_data[5]

        print("\n\tPREDICTION\n")
        # newdataset = self.df.drop(["Sex", "Varsta", "Zile_spitalizare", "zile_ATI", "Diag_pr_int", 'Analize_prim_set', "Comorbiditati", "Diag_pr_ext", "stare_externare", "forma_boala"], axis=0, inplace=False)
        newdataset = self.df.drop(
            range(1, self.df.shape[0]), axis=0, inplace=False)
        newdataset = newdataset.drop(["stare_externare", "forma_boala"], axis='columns')
        for column in newdataset.columns:
            newdataset[column] = 0

        newsex = 0
        if sex == 'Female':
            newsex = 0
        else:
            newsex = 1
        diag = self.comorbiditati[diag_init]

        analize_list = analize.split(" || ")
        for analiza in analize_list:
            analiza_name, rest = analiza.split(" - ", 1)
            result, ignore = rest.split(" ", 1)
            result = result.replace("<", "")
            analiza_name = analiza_name.replace(" ", "")
            try:
                result_int = float(result)
                try:
                    newdataset[analiza_name][0] = result_int
                except:
                    newdataset[analiza_name] = np.zeros(self.df.shape[0], dtype=int)
                    newdataset[analiza_name][0] = result_int
                    pd.to_numeric(newdataset[analiza_name])
            except:
                pass

        newdataset["Comorbiditati"][0] = self.df["Comorbiditati"].mean()
        newdataset["Varsta"][0] = age
        newdataset["Sex"][0] = newsex
        newdataset["Diag_pr_int"][0] = diag
        newdataset["Diag_pr_ext"][0] = 0
        newdataset["Zile_spitalizare"][0] = zile_spit
        newdataset["zile_ATI"][0] = zile_ati

        print(newdataset)

        return newdataset

    def get_dataset(self):
        return self.df

    def get_boala(self):
        return self.boala

    def describe(self):
        """
        Important information about the data:
            - the number of lines x columns
            - the first 5 records of the dataset
            - the type of data in every row
            - how many null values are in every row
            - mean, min, max and the 4 important quartiles of the data
            - how many unique values are in a column
        :return:
        """
        print(self.df.shape)
        print(self.df)
        print("\nData types and nonnull values")
        print(self.df.info())
        print("\nNull values in the dataset")
        print(self.df.isnull().sum())
        print("\nDescribed dataset")
        print(self.df.describe().T)
        print("\nUnique values")
        print(self.df.nunique())

    def replace_columns(self):
        """
        Removing redundant information.
        Replacing with Python NULL in the empty records.
        Making categorical data numerical. <- TO BE CHANGED, NOW HARDCODED
        :return:
        """
        self.df.drop(["AN", "precod", "FO", "Data_Examinare_Radiologie", "Radiologie", "rezultat_radiologie", "Proceduri", "Proceduri_Radiologie", "tip_externare", "unde_pleaca"], inplace=True, axis='columns')

        self.df.replace("NULL", np.NAN, inplace=True)
        self.df.replace("", np.NAN, inplace=True)
        self.df.replace("_", np.NAN, inplace=True)

        self.df.Sex.replace(("F", "M"), (0, 1), inplace=True)
        self.df.stare_externare.replace(("Vindecat", "Ameliorat", "Stationar", "AGRAVAT                                           ", "Decedat"), (0, 1, 2, 3, 4), inplace=True)
        self.df.forma_boala.replace(('1.USOARA', '2. MODERATA', '3.SEVERA', 'PERICLITAT TRANSFUZIONAL'), (1, 2, 3, np.NaN), inplace=True)
        self.df.forma_boala = self.df.forma_boala.fillna(self.df.forma_boala.median())

    def change_medicatie(self):
        """
        One Hot Encoding for the "Medicatie" column.
        :return:
        """
        d = {}
        indx = 0
        for record in self.df.Medicatie:
            med_list = str(record).split("||")
            for med in med_list:
                med = med.replace(" ", "")
                try:
                    self.df[med][indx] = 1
                except:
                    self.df[med] = np.zeros(self.df.shape[0], dtype=int)
                    self.df[med][indx] = 1
                    pd.to_numeric(self.df[med])
                d[med] = 1
            indx += 1
        for key, value in d.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Medicatie'], inplace=True, axis='columns')

    def analize_columns(self):
        """
        One Hot Encoding for the "Analize_prim_set" column.
        :return:
        """
        d = {}
        indx = 0
        for record in self.df.Analize_prim_set:
            if record is not np.NAN:
                record = record.replace("- HS * ", "")
                analize_list = record.split(" || ")
                for analiza in analize_list:
                    analiza_name, rest = analiza.split(" - ", 1)
                    result, ignore = rest.split(" ", 1)
                    result = result.replace("<", "")
                    analiza_name = analiza_name.replace(" ", "")
                    try:
                        result_int = float(result)
                        d[analiza_name] = 1
                        try:
                            self.df[analiza_name][indx] = result_int
                        except:
                            self.df[analiza_name] = np.zeros(self.df.shape[0], dtype=int)
                            self.df[analiza_name][indx] = result_int
                            pd.to_numeric(self.df[analiza_name])
                    except:
                        pass
            indx += 1
        for key, value in d.items():
            if self.df[key].sum() <= self.df.shape[0] * 0.2:
                self.df.drop([key], inplace=True, axis='columns')
        self.df.drop(['Analize_prim_set'], inplace=True, axis='columns')

    def comorbiditati_columns(self):
        """
        Mambo Jumbo <- TO BE CHANGED, WE DON'T KNOW WHAT IS THIS FOR NOW
        :returns: Dictionary[illness code: weight]
        """
        self.boala = pd.DataFrame()
        self.boala = self.boala_dataset()

        weight = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        forma_weight = {0: -1, 1: -0.5, 2: 0.25, 3: 0.5, 4: 1}
        total_count = self.df.stare_externare.value_counts().sum()
        count_forma = pd.DataFrame(self.df.stare_externare.value_counts()).to_dict()['stare_externare']
        for i in range(0, 5):
            weight[i] = forma_weight[i] * (1 - count_forma[i] / total_count)

        col_names = self.boala.index
        forma = {0: 'Vindecat', 1: 'Ameliorat', 2: 'Stationar', 3: 'Agravat', 4: 'Decedat'}
        d = {}
        for names in col_names:
            d[names] = 0
            for i in range(0, 5):
                d[names] += self.boala[forma[i]][names] * weight[i]

        indx = 0
        for row in self.df.Comorbiditati:
            if row is not np.NaN:
                comb_list = row.split(",")
                comb_weight = 0
                for comb in comb_list:
                    comb = comb.split(" ", 1)[0]
                    if comb in d:
                        comb_weight += d[comb]
                self.df["Comorbiditati"][indx] = float(comb_weight)
            else:
                self.df["Comorbiditati"][indx] = 0
            indx += 1

        self.df["Comorbiditati"].replace(0, np.NAN, inplace=True)
        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float).interpolate(method='polynomial', order=2)

        self.df["Comorbiditati"] = self.df["Comorbiditati"].astype(float)

        return d

    def diagnos_columns(self):
        """
        Tied to the function above => CASCADE ON EDIT ABOVE FUNCTION
        :return:
        """
        indx = 0
        for row_int in self.df.Diag_pr_int:
            if row_int is not np.NaN:
                try:
                    self.df["Diag_pr_int"][indx] = self.comorbiditati[row_int]
                except:
                    self.df["Diag_pr_int"][indx] = np.NAN
            indx += 1
        self.df.Diag_pr_int = self.df.Diag_pr_int.astype(float).interpolate(method='polynomial', order=2)
        self.df.Diag_pr_int = self.df.Diag_pr_int.fillna(method='bfill')
        self.df["Diag_pr_int"] = self.df["Diag_pr_int"].astype(float)

        indx = 0
        for row_ext in self.df.Diag_pr_ext:
            if row_ext is not np.NaN:
                try:
                    self.df["Diag_pr_ext"][indx] = self.comorbiditati[row_ext]
                except:
                    self.df["Diag_pr_ext"][indx] = np.NAN
            indx += 1
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].astype(float).interpolate(method='polynomial', order=2)
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].fillna(method='bfill')
        self.df["Diag_pr_ext"] = self.df["Diag_pr_ext"].astype(float)

    def boala_dataset(self):
        """
        Creates a dictionary with every illness and how many people had each type of severity.
        :returns: DataFrame
        """
        self.com_ext = self.df[["Comorbiditati", "stare_externare"]]
        d = {}
        g = open("text-comorbiditati.txt", "r")
        r = csv.reader(g)
        for row in r:
            # count, vindecat, ameliorat, stationar, agravat, decedat
            ds = row[0].split(" ", 1)[0]
            d[ds] = [0, 0, 0, 0, 0, 0]
            for cr, ext in self.com_ext.itertuples(index=False):
                if type(cr) is str and row[0] in cr:
                    d[ds][0] = d[ds][0] + 1
                    d[ds][int(ext) + 1] = d[ds][int(ext) + 1] + 1
        dictr = {}
        for key, value in d.items():
            dis = key.split(" ", 1)
            dictr[dis[0]] = value[5]
        g.close()

        self.boala = pd.DataFrame(d)
        self.boala = self.boala.transpose()
        self.boala.rename(
            columns={0: 'Count', 1: 'Vindecat', 2: 'Ameliorat', 3: 'Stationar', 4: 'Agravat', 5: 'Decedat'},
            inplace=True, errors="raise")
        return self.boala

    def scale_data(self):
        pass

    '''
    Plotting functions
    '''
    def plt_data_distribution(self):
        sns.set_palette("Purples_r")

        plt.subplots(figsize=(13, 5))
        sns.countplot(data=self.df, x='Zile_spitalizare')
        plt.xticks(rotation=90)
        plt.xlabel("Zile spitalizare")
        plt.title("Distributia zilelor de spitalizare")
        plt.show()

        sns.kdeplot(self.df.stare_externare, data=self.df, shade=True, hue='Sex', label=['femeie', 'barbat'])
        plt.xlabel("Stare de externare")
        plt.ylabel("Count")
        plt.title("Distributia starilor de externare in functie de sex")
        plt.show()

        sns.kdeplot(self.df.Varsta, data=self.df, shade=True, hue='Sex', label=['femeie', 'barbat'])
        plt.title("Distributia varstei in functie de sex")
        plt.show()

        sns.kdeplot(self.df.Zile_spitalizare, data=self.df, shade=True, hue='Sex', label=['femeie', 'barbat'])
        plt.xlabel("Zile de spitalizare")
        plt.ylabel("Count")
        plt.title("Distributia starilor de externare in functie de sex")
        plt.show()

        plt.subplots(figsize=(12, 5))
        sns.distplot(self.df["Varsta"], bins=25, kde=True, rug=False)
        plt.title('Distributia varstei')
        plt.show()

        plt.subplots(figsize=(12, 5))
        sns.distplot(self.df["Zile_spitalizare"], bins=70, kde=True, rug=False)
        plt.title('Distributia zilelor de spitalizare')
        plt.show()

        sns.jointplot(x='stare_externare', y='Zile_spitalizare', data=self.df)
        plt.xlabel("0-Vindecat, 1-Ameliorat, 2-Stationar, 3-Agravat, 4-Decedat")
        plt.ylabel("Zile de spitalizare")
        plt.show()

        with sns.axes_style('white'):
            sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='hex')
            plt.show()

        sns.stripplot(data=self.df, x='stare_externare', y='Varsta', jitter=True, marker='.')
        plt.xlabel("Stare externare")
        plt.title("Cate valori sunt pentru fiecare stare de externare, distribuite pe varsta")
        plt.show()

        f = sns.FacetGrid(self.df, col="stare_externare")
        f.map(plt.hist, "Zile_spitalizare")
        plt.xlim(0, 45)
        plt.show()

        with sns.color_palette("Purples"):
            f = sns.FacetGrid(self.df, col="forma_boala", hue="stare_externare")
            f.map(plt.scatter, "Zile_spitalizare", "zile_ATI", alpha=0.5, marker='.')
            f.add_legend()
            plt.xlim(0, 70)
            plt.show()

        g = sns.PairGrid(self.df, vars=["Zile_spitalizare", "zile_ATI", "Varsta"], hue="forma_boala")
        # g.map(plt.scatter)
        # g.map_diag(sns.histplot)
        # g.map_offdiag(sns.scatterplot)
        g.map_upper(sns.scatterplot, size=self.df["Sex"], alpha=0.5)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        g.add_legend(title="", adjust_subtitles=True)
        plt.show()

    def plt_stare_externare(self):
        sns.set_palette("Purples_r")

        data_vindecat = self.df[self.df["stare_externare"] == 0]
        data_decedat = self.df[self.df["stare_externare"] == 4]
        data_ameliorat = self.df[self.df["stare_externare"] == 1]
        varsta_vindecat = data_vindecat["Varsta"]
        varsta_decedat = data_decedat["Varsta"]
        varsta_ameliorat = data_ameliorat["Varsta"]
        spitalizare_vindecat = data_vindecat["Zile_spitalizare"]
        spitalizare_decedat = data_decedat["Zile_spitalizare"]
        spitalizare_ameliorat = data_ameliorat['Zile_spitalizare']
        ati_vindecat = data_vindecat["zile_ATI"]
        ati_ameliorat = data_ameliorat["zile_ATI"]
        ati_decedat = data_decedat["zile_ATI"]

        plt.plot(varsta_vindecat, spitalizare_vindecat, ".")
        plt.plot(varsta_ameliorat, spitalizare_ameliorat, ".", alpha=0.5)
        plt.plot(varsta_decedat, spitalizare_decedat, ".", alpha=0.5)
        plt.xlabel('Varsta')
        plt.ylabel('Zile de spitalizare')
        plt.legend(["Vindecat", "Ameliorat", "Decedat"])
        plt.title('Zilele de spitalizare pe varsta in functie de starea de externare')
        plt.show()

        plt.plot(varsta_vindecat, ati_vindecat, ".")
        plt.plot(varsta_ameliorat, ati_ameliorat, ".")
        plt.plot(varsta_decedat, ati_decedat, ".")
        plt.xlabel('Varsta')
        plt.ylabel('Zile de petrecute la ATI')
        plt.legend(["Vindecat", "Ameliorat", "Decedat"])
        plt.show()

        fig = plt.figure(figsize=(9, 6))
        sns.histplot(data_decedat, x=data_decedat["Varsta"], y=data_decedat["Zile_spitalizare"], cmap='OrRd',
                     kde=True, label='Decedat', bins=45)
        sns.histplot(data_vindecat, x=data_vindecat["Varsta"], y=data_vindecat["Zile_spitalizare"], cmap='BuGn',
                     kde=True, label='Vindecat', bins=45, alpha=0.5)
        plt.ylabel("Zile de spitalizare")
        patch1 = mpatches.Patch(color='green', label='Vindecat')
        patch2 = mpatches.Patch(color='orange', label='Decedat')
        plt.legend(handles=[patch1, patch2])
        plt.title("Zilele de spitalizare in functie de varsta")
        plt.show()


'''
plan:
    create instance of prediction model at the beginning of the program
    -> data is cleaned and scaled
    -> model is trained or the existing model is loaded
    GET the JSON input
    initialize a Medical_record class with the given values
    call the predict function over the given data
    -> predict the output
    -> save the data in the csv
    -> return (POST) the outcome
    
    (-> return (POST) a visualization of the model or of the result)
'''