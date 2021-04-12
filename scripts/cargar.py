#Importar los paquetes necesarios
import tensorflow as tf
from keras.models import load_model

#Funcion para leer el modelo h5
def leerModelo():

    FILENAME_MODEL_TO_LOAD = "covid19_model_full.h5"
    MODEL_PATH = "../../model"

    #Cargar el modelo
    loaded_model = load_model(MODEL_PATH + "/" + FILENAME_MODEL_TO_LOAD)
    print("El modelo se ha cargado: ", loaded_model)

    graph = tf.get_default_graph()
    return loaded_model, graph