import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from time import time
import pandas as pd


def splitter(inputs, outputs, random_state):
    """
    Divide un conjunto de datos en conjuntos de entrenamiento, validación y test.

    Args:
        inputs (numpy.ndarray): datos de entrada.
        outputs (numpy.ndarray): datos de salida.
        random_state (int): semilla aleatoria.

    Returns:
        X_train (numpy.ndarray): entrada para el conjunto de entrenamiento.
        y_train (numpy.ndarray): salida para el conjunto de entrenamiento.
        X_val (numpy.ndarray): entrada para el conjunto de validación.
        y_val (numpy.ndarray): salida para el conjunto de validación.
        X_test (numpy.ndarray): entrada para el conjunto de test.
        y_test (numpy.ndarray): salida para el conjunto de test.
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(inputs, outputs, test_size=0.1, stratify=outputs, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, stratify=y_train_val, random_state=random_state)

    return X_train, y_train, X_val, y_val, X_test, y_test 



def get_metrics(real, pred, labels=[0, 1, 2], labels_names=['elephant', 'rhino', 'other']):
    """
    Obtiene las métricas relevantes para el resultado de la aproximación.

    Args:
        real (numpy.ndarray): clases reales.
        pred (numpy.ndarray): clases predichas.
        labels (list): códigos de las clases.
        labels_names (list): etiquetas de las clases.

    Returns:
        metrics (dict): diccionario con las métricas.
    """
    
    accuracy = accuracy_score(real, pred)
    table_data = {'precision': precision_score(real, pred, average=None, labels=labels), 
                  'recall': recall_score(real, pred, average=None, labels=labels), 
                  'f1': f1_score(real, pred, average=None, labels=labels)}
     
    df = pd.DataFrame(table_data, index=labels_names)
    df.loc['mean'] = df.mean()
    df = df.round(2)

    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(real, pred, labels=labels), display_labels=labels_names)
    fig, ax = plt.subplots(figsize=(3, 5))
    cm.plot(ax=ax, colorbar=False, cmap='Blues')

    print(F'### TABLA CON MÉTRICAS POR CLASE (accuracy total: {accuracy})')
    display(df)
    print(F'\n### MATRIZ DE CONFUSIÓN')
    plt.show()

    metrics = {'accuracy': accuracy, 'table': df, 'cm': cm}
    return metrics



def metrics_table(summary):
    """
    Muestra una tabla con las métricas de las aproximaciones.

    Args:
        summary (dict): diccionario con los resultados de las aproximaciones.

    Returns:
        df (DataFrame): tabla con todos los resultados.
    """
    df = pd.DataFrame()
    aproaches = list(summary.keys())

    for i in range(len(aproaches)):
        df2 = pd.DataFrame({
            'aproximación': [aproaches[i]],
            'ciclos': [summary[aproaches[i]]['cycles']],
            'duración (s)': [summary[aproaches[i]]['duration']],
            'accuracy': [summary[aproaches[i]]['metrics']['accuracy']],
            'precision (mean)': [summary[aproaches[i]]['metrics']['table'].iloc[-1]['precision']],
            'recall (mean)': [summary[aproaches[i]]['metrics']['table'].iloc[-1]['recall']],
            'f1 (mean)': [summary[aproaches[i]]['metrics']['table'].iloc[-1]['f1']]
        })
        df = pd.concat([df, df2], ignore_index=True)

    return df



def plot_all_cm(summary):
    """
    Muestra las matrices de confusión del resultado de las aproximaciones.

    Args:
        summary (dict): diccionario con los resultados de las aproximaciones.

    Returns:
        None
    """
    
    aproaches = list(summary.keys())
    cols = 4
    rows = int(np.ceil(len(aproaches) / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axs = axs.flatten()
    
    for i, aproach in enumerate(aproaches):
        cm = summary[aproach]['metrics']['cm']
        cm.plot(ax=axs[i], colorbar=False, cmap='Blues')
        axs[i].set_title(f'aproach_{i}')
    
    for j in range(len(aproaches), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()



def plot_animals(animals, preprocessed, samples=[42,230,72,155], figsize=(8, 8)):
    """
    Muestra las imágenes antes y después del preprocesado.

    Args:
        animals (np.array): imágenes de entrada en color.
        preprocessed (np.array): imágenes preprocesadas.
        sample (list): índices de las imágenes a mostrar.
        figsize (tupe): tamaño del plot.

    Returns:
        None
    """    

    channels = int(preprocessed.shape[1]/(300*300))
    
    fig, axs = plt.subplots(len(samples), channels+1, figsize=figsize)

    for i, sample in enumerate(samples):

        axs[i,0].imshow(animals[sample], cmap='gray')
        axs[0,0].set_title('Original')
        axs[i,0].axis("off")

        for j in range(channels):
            axs[i,j+1].imshow(preprocessed[sample].reshape(300,300, channels)[:,:,j], cmap='gray')
            axs[0,j+1].set_title(f'Preprocessed (dim {j+1})')
            axs[i,j+1].axis("off")

    plt.tight_layout()
    plt.show()



class XGBModel:

    def __init__(self):
        '''
        Inicializa el modelo.
        '''

        self.xgb = xgb.XGBClassifier(objective='multi:softprob', early_stopping_rounds=10)


    def train(self, X_train, y_train, X_val, y_val):
        '''
        Entrena el modelo.

        Args:
            X_train (numpy.ndarray): entrada para el conjunto de entrenamiento.
            y_train (numpy.ndarray): salida para el conjunto de entrenamiento.
            X_val (numpy.ndarray): entrada para el conjunto de validación.
            y_val (numpy.ndarray): salida para el conjunto de validación.

        Returns:
            cycles(int): número de ciclos de entrenamiento.
            duration(float): duración del entrenamiento en segundos.
        '''

        start = time()
        self.xgb.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)], verbose=False)
        end = time()
        duration = round(end - start, 2)
        
        results = self.xgb.evals_result()
        cycles = len(results['validation_0']['mlogloss'])

        plt.plot(results['validation_0']['mlogloss'], label='train')
        plt.plot(results['validation_1']['mlogloss'], label='val')
        plt.legend()
        plt.show()

        return cycles, duration


    def predict(self, X_test):
        '''
        Predice la clase a la que pertenece la imagen.

        Args:
            X_test (numpy.ndarray): entrada para el conjunto de test.

        Returns:
            predictions (numpy.ndarray): clase predicha para el conjunto de test.
        '''
        
        return self.xgb.predict(X_test)


