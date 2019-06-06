import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from inspect import signature

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

