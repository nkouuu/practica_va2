import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import seaborn as sn


class Graphics:

    def accuracy_graphic(self, accuracy, title):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = 'Aciertos', 'Fallos'
        sizes = [accuracy, 100 - accuracy]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Resultados {title}')
        plt.show()

    def conf_matrix(self, classifier_result, labels):
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
        df_confusion = confusion_matrix(labels, classifier_result, labels=list(range(0, 42)))
        df_cm = pd.DataFrame(df_confusion, index = [i for i in range(0, len(df_confusion))], columns = [i for i in range(0, len(df_confusion))])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True)
        plt.show()

    def get_f1_score(self, samples, labels):
        precision = precision_score(labels, samples, average='micro')
        recall = recall_score(labels, samples, average='micro')
        f1score = f1_score(labels, samples, average='micro')
        print(f'F1_score test samples: {f1score}')
