import model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import missingno as msno
from scipy.stats import pearsonr, spearmanr
import csv


class DataProcessing:

    def __init__(self, dataset_name):
        self.dataset = pd.read_csv(dataset_name, parse_dates=True)
        self.df = pd.DataFrame(self.dataset)

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
        # self.data = [[self.__preg, self.__gluc, self.__blpr, self.__skth, self.__insu, self.__bmi, self.__pedi, self.__age]]

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

    '''
    Plotting functions
    '''
    def plt_data_distribution(self):
        sns.displot(self.df.Zile_spitalizare, kde=True, rug=True, bins=100)
        # sns.displot(self.df.zile_ATI, kde=True)
        plt.show()

        sns.kdeplot(self.df.stare_externare, data=self.df, shade=True, color='b', hue='Sex')
        plt.show()

        sns.kdeplot(self.df.Varsta, data=self.df, shade=True, color='r', hue='Sex')
        plt.show()

        sns.jointplot(x='stare_externare', y='Zile_spitalizare', data=self.df)
        plt.xlabel("0-Vindecat, 1-Ameliorat, 2-Stationar, 3-Agravat, 4-Decedat")
        plt.ylabel("Zile de spitalizare")
        plt.show()

        with sns.axes_style('white'):
            sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='hex', color='k')
            plt.show()

    def plt_stare_externare(self):
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
        plt.plot(varsta_ameliorat, spitalizare_ameliorat, ".")
        plt.plot(varsta_decedat, spitalizare_decedat, ".")
        plt.xlabel('Varsta')
        plt.ylabel('Zile de spitalizare')
        plt.legend(["Vindecat", "Ameliorat", "Decedat"])
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
        plt.show()

    def plt2(self):
        sns.displot(self.df.Zile_spitalizare, kde=True, rug=True, bins=100)
        # sns.displot(self.df.zile_ATI, kde=True)
        plt.show()

        sns.kdeplot(self.df.stare_externare, shade=True, color='b')
        plt.show()

        # sns.kdeplot(self.df.Varsta, shade=True, color='r')
        # plt.show()

        sns.kdeplot(self.df.Varsta, data=self.df, shade=True, color='r', hue='Sex')
        plt.show()

        sns.histplot(self.df, x=self.df.stare_externare, hue="Sex", palette="ch:s=.25,rot=-.25", bins=35, alpha=0.5)
        plt.xlabel("Stare externare")
        plt.show()

        sns.jointplot(x='stare_externare', y='Zile_spitalizare', data=self.df)
        plt.xlabel("0-Vindecat, 1-Ameliorat, 2-Stationar, 3-Agravat, 4-Decedat")
        plt.ylabel("Zile de spitalizare")
        plt.show()

        with sns.axes_style('white'):
            sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='hex', color='k')
            plt.show()

        with sns.axes_style('white'):
            with sns.color_palette("Purples"):
                sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='kde', shade=True)
                plt.show()

        with sns.axes_style('white'):
            with sns.color_palette("Purples"):
                sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, kind='kde', shade=True,
                              hue='forma_boala', fill=True)
                plt.show()

        with sns.axes_style('white'):
            sns.jointplot(x="forma_boala", y='stare_externare', data=self.df, hue='Sex')
            plt.show()

    def plt3(self):
        sns.pairplot(self.df, diag_kind="kde")
        plt.show()

        sns.pairplot(self.df, vars=["Varsta", "Zile_spitalizare", "zile_ATI"], diag_kind="kde",
                     hue='stare_externare', palette='light:#3A9', corner=True)
        plt.show()

        corr = self.df.corr()
        plt.subplots(figsize=(12, 12))
        sns.heatmap(corr, vmax=1, annot=True, fmt='.2f', cmap="YlOrRd_r")
        plt.show()

        sns.lmplot(x='zile_ATI', y='Zile_spitalizare', data=self.df, aspect=2)
        plt.show()

        sns.lmplot(data=self.df, x='stare_externare', y='Zile_spitalizare', x_jitter=.1, height=9,
                   x_estimator=np.mean)
        plt.legend(["0-Vindecat, 1-Ameliorat, 2-Stationar, 3-Agravat, 4-Decedat"])
        plt.xlim(-1, 5)
        plt.ylim(0, 22.5)
        plt.show()

        sns.lmplot(data=self.df, x='Zile_spitalizare', y='zile_ATI', col='forma_boala', hue='Sex')
        plt.show()

        sns.regplot(data=self.df, x='stare_externare', y='Zile_spitalizare')
        plt.xlim(-1, 5)
        plt.ylim(-0.5, 90)
        plt.show()

    def plt4(self):
        sns.stripplot(data=self.df, x='stare_externare', y='Varsta', jitter=True)
        plt.show()

        sns.swarmplot(data=self.df, x='stare_externare', y='Varsta', marker='.')
        plt.show()

        sns.catplot(data=self.df, x='forma_boala', y='Varsta', kind="violin", split=True, hue='Sex')
        plt.show()

        # sns.catplot(data=self.df, x='forma_boala', y='Varsta', kind="bar", hue='Sex')
        # plt.show()

        plt.subplots(figsize=(15, 5))
        sns.swarmplot(data=self.df, y='forma_boala', x='zile_ATI')
        plt.show()

        sns.boxplot(data=self.df, x='stare_externare', y='Zile_spitalizare')
        plt.show()

        sns.violinplot(data=self.df, x='stare_externare', y='Zile_spitalizare', inner='stick')
        plt.show()

        # plt.subplots(figsize=(13, 5))
        # sns.barplot(data=self.df, x='Varsta', y='zile_ATI', hue='forma_boala')
        # plt.xlim(min(self.df.Varsta), max(self.df.Varsta))
        # plt.xticks(rotation=90)
        # plt.show()

        plt.subplots(figsize=(13, 5))
        sns.countplot(data=self.df, x='Zile_spitalizare')
        plt.xticks(rotation=90)
        plt.show()

        sns.pointplot(data=self.df, x='stare_externare', y='forma_boala', hue='Sex')
        plt.show()

        sns.factorplot(data=self.df, x='stare_externare', y='forma_boala', hue='Sex', kind='swarm', size=5,
                       aspect=2)
        plt.show()

    def plt5(self):
        sns.pairplot(self.boala)
        plt.show()

        corrb = self.boala.corr()
        plt.subplots(figsize=(12, 12))
        sns.heatmap(corrb, vmax=1, annot=True, fmt='.2f', cmap="YlOrRd_r")
        plt.show()

        plt.subplots(figsize=(15, 10))
        msno.bar(self.df)
        plt.show()

        pf = self.df
        del pf["Sex"]
        del pf["Diag_pr_int"]
        del pf["forma_boala"]
        corr = pf.corr()
        plt.subplots(figsize=(12, 12))
        sns.heatmap(corr, vmax=1, annot=True, fmt='.2f', cmap="YlOrRd_r")
        plt.show()

    def plt6(self):
        f = sns.FacetGrid(self.df, col="stare_externare")
        f.map(plt.hist, "Zile_spitalizare")
        plt.xlim(0, 45)
        plt.show()

        f = sns.FacetGrid(self.df, col="forma_boala", hue="stare_externare")
        f.map(plt.scatter, "Zile_spitalizare", "zile_ATI", alpha=0.7)
        f.add_legend()
        plt.xlim(0, 100)
        plt.show()

        h = {0: 'blue', 1: 'green'}
        i = {"marker": ["^", "v"]}
        f = sns.FacetGrid(self.df, row="stare_externare", col="forma_boala", hue="Sex", margin_titles=True,
                          gridspec_kws={"width_ratios": [2, 4, 4]}, palette=h, hue_kws=i)
        f.map(sns.regplot, "Zile_spitalizare", "zile_ATI", fit_reg=False)
        f.add_legend()
        plt.show()

        f = sns.FacetGrid(self.df, col="stare_externare", col_wrap=2)
        f.map(sns.barplot, "Sex", "Zile_spitalizare", color="#9FC2E9", edgecolor="blue", lw=.5)
        f.fig.subplots_adjust(wspace=1, hspace=0.5)
        plt.show()

        f = sns.FacetGrid(self.df, col="stare_externare", col_wrap=2)
        f.map(sns.barplot, "forma_boala", "Zile_spitalizare", color="#9FC2E9", edgecolor="blue", lw=.5)
        f.set_axis_labels("1-Usor, 2-Moderat, 3-Sever", "Zile petrecute in spital")
        f.set(yticks=[0, 5, 10, 20])
        f.set(xlim=(-1, 3), ylim=(0, 20))
        plt.show()

    def plt7(self):
        g = sns.PairGrid(self.df, vars=["Zile_spitalizare", "zile_ATI", "Varsta"], hue="forma_boala")
        # g.map(plt.scatter)
        # g.map_diag(sns.histplot)
        # g.map_offdiag(sns.scatterplot)
        g.map_upper(sns.scatterplot, size=self.df["Sex"])
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.kdeplot)
        g.add_legend(title="", adjust_subtitles=True)
        plt.show()

        g = sns.PairGrid(self.df, x_vars=["Zile_spitalizare", "zile_ATI"], y_vars=["Varsta"], hue="forma_boala",
                         hue_kws={"marker": ["^", "v", "^"]})
        g.map(plt.scatter)
        g.add_legend(title="", adjust_subtitles=True)
        plt.show()

    def plt8(self):
        plt.subplots(figsize=(15, 5))
        sns.distplot(self.df["Varsta"], bins=20, kde=True, rug=False)
        plt.show()

        sns.jointplot(x="Varsta", y='Zile_spitalizare', data=self.df, size=8)
        plt.show()

        sns.pairplot(data=self.df, size=5, aspect=0.5, x_vars=["Zile_spitalizare", "zile_ATI"],
                     y_vars=["forma_boala", "stare_externare"], kind="reg", hue='Sex')
        plt.show()

    def plt9(self):
        sns.set()
        sns.set_style("darkgrid")
        sns.set_style("ticks")
        sns.displot(self.df["Zile_spitalizare"], kde=True)
        # sns.despine(left=True, right=False, bottom=True, top=False)
        sns.despine(offset=5, trim=False)
        plt.xlabel("Zile la spitalizare")
        plt.ylabel("")
        plt.show()

        sns.set_palette("Greens_d")
        sns.displot(self.df["Zile_spitalizare"], kde=True)
        plt.show()

        sns.palplot(sns.color_palette("hls", 12))
        plt.show()

        sns.palplot(sns.hls_palette(10, h=0.5, l=0.8, s=0.6))
        plt.show()

        sns.palplot(sns.color_palette("Blues_d"))
        plt.show()

        cmap = sns.light_palette("green", as_cmap=True, reverse=True)
        sns.kdeplot(self.df["Varsta"], self.df["Zile_spitalizare"], cmap=cmap)
        plt.ylim(-5, 50)
        plt.show()

        sns.set_style("dark")
        sns.set_context("talk")
        f, ax = plt.subplots(figsize=(10, 8))
        cmap = sns.cubehelix_palette(as_cmap=True, dark=1, light=0, reverse=True)
        sns.kdeplot(self.df["zile_ATI"], self.df["Zile_spitalizare"], ax=ax, cmap=cmap, shade=True)
        plt.ylim(-5, 50)
        plt.xlim(-7, 30)
        plt.xlabel("Zile pertecute la ATI")
        plt.ylabel("Zile spitalizare")
        plt.show()

    def plt10(self):
        self.df[['Zile_spitalizare', 'zile_ATI']].boxplot(figsize=(12, 8))
        plt.title("Zile de spitalizare si zile petrecute la ATI")
        plt.show()

        self.df.boxplot(column=['Varsta'], by=['stare_externare'], figsize=(12, 8), color='red')
        plt.show()

        forma_df = self.df.groupby('forma_boala', as_index=False).mean().round(3)
        print(forma_df)
        forma_df.plot.bar(x='Zile_spitalizare', y='Varsta', figsize=(12, 14), color=['C0', 'C1', 'C3'],
                          legend=False)
        plt.show()

    def plt11(self):
        print(np.corrcoef(self.df.Zile_spitalizare, self.df.zile_ATI))
        print(pearsonr(self.df.Zile_spitalizare, self.df.zile_ATI))
        print(spearmanr(self.df.Zile_spitalizare, self.df.zile_ATI))

        plt.figure(figsize=(12, 8))
        plt.scatter(self.df.Zile_spitalizare, self.df.zile_ATI, color='C1')
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