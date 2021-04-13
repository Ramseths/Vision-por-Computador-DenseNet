#Importar los paquetes necesarios
import tensorflow as tf
from keras.models import load_model

#Funcion para leer el modelo h5
def leerModelo():

    FILENAME_MODEL_TO_LOAD = "ramseths_densenet.h5"
    MODEL_PATH = "../../modelo"

    #Cargar el modelo
    modelo_cargado = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("El modelo se ha cargado: ", modelo_cargado)

    graph = tf.get_default_graph()
    return modelo_cargado, graph