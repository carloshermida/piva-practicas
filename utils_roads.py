import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
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
        sample_weights (numpy.ndarray): pesos de las muestras del conjunto de entrenamiento.
        test_indices (numpy.ndarray): índices de las muestras del conjunto de test.
    """

    image_indices = np.arange(20)

    train_val_indices, test_indices = train_test_split(image_indices, test_size=2/20, random_state=random_state)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=2/18, random_state=random_state)

    X_train, y_train = inputs[train_indices], outputs[train_indices]
    X_val, y_val = inputs[val_indices], outputs[val_indices]
    X_test, y_test = inputs[test_indices], outputs[test_indices]

    channels = X_train.shape[2] if len(X_train.shape) > 2 else 1
    
    X_train, X_test, X_val = [data.reshape(-1, channels) for data in (X_train, X_test, X_val)]
    y_train, y_test, y_val = [data.flatten() for data in (y_train, y_test, y_val)]

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test, sample_weights, test_indices



def get_metrics(real, pred):
    """
    Obtiene las métricas relevantes para el resultado de la aproximación

    Args:
        real (numpy.ndarray): ground truth.
        pred (numpy.ndarray): resultado final de la aproximación.

    Returns:
        metrics (dict): diccionario con las métricas.
    """

    metrics = {}
    metrics['accuracy'] = round(accuracy_score(real, pred), 4)
    metrics['precision'] = round(precision_score(real, pred, zero_division=1), 4)
    metrics['recall'] = round(recall_score(real, pred, zero_division=1), 4)
    metrics['f1'] = round(f1_score(real, pred, zero_division=1), 4)
    metrics['jaccard'] = round(jaccard_score(real, pred), 4)

    return metrics



def plot_all_approaches(sat, summary, gt, test_indices):
    """
    Muestra gráficamente el resultado de las aproximaciones.

    Args:
        sat (numpy.ndarray): imágenes de entrada.
        summary (dict): diccionario con los resultados de las aproximaciones.
        gt (numpy.ndarray): ground truth.
        test_indices (numpy.ndarray): índices de las muestras del conjunto de test.

    Returns:
        None
    """

    aproaches = list(summary.keys())

    for i in range(len(test_indices)):

        fig, axs = plt.subplots(1, len(aproaches)+2, figsize=(15, 5))

        axs[0].imshow(sat[test_indices][i])
        axs[0].set_title('sat')
        axs[0].axis('off')

        for j in range(len(aproaches)):
            axs[j+1].imshow(summary[aproaches[j]]['image'][i].reshape(1500,1500), cmap='gray')
            axs[j+1].set_title(f'aproach_{j}')
            axs[j+1].axis('off')

        axs[-1].imshow(gt[test_indices][i].reshape(1500,1500), cmap='gray')
        axs[-1].set_title('gt')
        axs[-1].axis('off')

        plt.tight_layout()
        plt.show()



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
            'treshold': [summary[aproaches[i]]['treshold']],
            'accuracy': [summary[aproaches[i]]['metrics']['accuracy']],
            'precision': [summary[aproaches[i]]['metrics']['precision']],
            'recall': [summary[aproaches[i]]['metrics']['recall']],
            'f1': [summary[aproaches[i]]['metrics']['f1']],
            'jaccard': [summary[aproaches[i]]['metrics']['jaccard']]
        })
        df = pd.concat([df, df2], ignore_index=True)

    return df



def plot_approach_result(sat, preprocess, prediction, postprocess, gt, test_indices):
    """
    Muestra gráficamente el proceso por el que pasa la aproximación.

    Args:
        sat (numpy.ndarray): imágenes de entrada.
        preprocess (numpy.ndarray): entrada preprocesada.
        prediction (numpy.ndarray): salida predicha por el modelo para el conjunto de test.
        postprocess (numpy.ndarray): resultado final de la aproximación.
        gt (numpy.ndarray): ground truth.
        test_indices (numpy.ndarray): índices de las muestras del conjunto de test.

    Returns:
        None
    """
    for i in range(len(test_indices)):

        fig = plt.figure(figsize=(23, 5))
        subfigs = fig.subfigures(1, 5)

        axs0 = subfigs[0].subplots(1,1)
        axs0.imshow(sat[test_indices][i])
        axs0.set_title('sat')
        axs0.axis('off')

        if len(preprocess.shape) > 2:
            sub_dim = preprocess.shape[2] 
            n = int(np.ceil(np.sqrt(sub_dim)))
            axs1 = subfigs[1].subplots(n,n)
            for j in range(n*n):
                ax = axs1.flatten()[j]
                if j < sub_dim:
                    ax.imshow(preprocess[test_indices][i][:,j].reshape(1500,1500), cmap='gray')
                    ax.set_title(f'preprocess dim {j+1}')
                ax.axis('off')
        else:
            axs1 = subfigs[1].subplots(1,1)
            axs1.imshow(preprocess[test_indices][i].reshape(1500,1500), cmap='gray')
            axs1.set_title('preprocess')
            axs1.axis('off')

        axs2 = subfigs[2].subplots(1,1)
        axs2.imshow(prediction[i].reshape(1500,1500), cmap='gray')
        axs2.set_title('prediction')
        axs2.axis('off')

        axs3 = subfigs[3].subplots(1,1)
        axs3.imshow(postprocess[i].reshape(1500,1500), cmap='gray')
        axs3.set_title('postprocess')
        axs3.axis('off')

        axs4 = subfigs[4].subplots(1,1)
        axs4.imshow(gt[test_indices][i].reshape(1500,1500), cmap='gray')
        axs4.set_title('gt')
        axs4.axis('off')
    
        plt.tight_layout()
        plt.show()



class XGBModel:

    def __init__(self):
        '''
        Inicializa el modelo.
        '''

        self.xgb = xgb.XGBClassifier(objective='binary:logistic', early_stopping_rounds=10)


    def train(self, X_train, y_train, X_val, y_val, sample_weights):
        '''
        Entrena el modelo.

        Args:
            X_train (numpy.ndarray): entrada para el conjunto de entrenamiento.
            y_train (numpy.ndarray): salida para el conjunto de entrenamiento.
            X_val (numpy.ndarray): entrada para el conjunto de validación.
            y_val (numpy.ndarray): salida para el conjunto de validación.
            sample_weights (numpy.ndarray): pesos de las muestras del conjunto de entrenamiento.

        Returns:
            cycles(int): número de ciclos de entrenamiento.
            duration(float): duración del entrenamiento en segundos.
        '''

        start = time()
        self.xgb.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_train, y_train),(X_val, y_val)], verbose=False)
        end = time()
        duration = round(end - start, 2)
        
        results = self.xgb.evals_result()
        cycles = len(results['validation_0']['logloss'])

        plt.plot(results['validation_0']['logloss'], label='train')
        plt.plot(results['validation_1']['logloss'], label='val')
        plt.legend()
        plt.show()

        return cycles, duration


    def predict(self, X_test, treshold=None):
        '''
        Predice que píxeles son carretera en una imagen.

        Args:
            X_test (numpy.ndarray): entrada para el conjunto de test.
            treshold (float): umbral para considerar carretera o no carretera. 
                              Si None, se devuelven las probabilidades.

        Returns:
            predictions (numpy.ndarray): salida predicha para el conjunto de test.
        '''
        if treshold:
            predictions_concatenated = (self.xgb.predict_proba(X_test)[:,1] >= treshold).astype('int') # dim 1, prob de ser carretera
        else:
            predictions_concatenated = self.xgb.predict_proba(X_test)[:,1]

        i = 0
        predictions = np.empty((0,2250000), int)
        
        while i < len(predictions_concatenated):
            predictions = np.vstack([predictions, predictions_concatenated[i:i+2250000]]) 
            i += 2250000
        
        fig, axs = plt.subplots(1, len(predictions), figsize=(7, 7))
        for i in range(len(predictions)):
            axs[i].imshow(predictions[i].reshape(1500, 1500), cmap='gray')
            axs[i].set_title(f'Imagen test {i+1} (treshold={treshold})')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

        return predictions


