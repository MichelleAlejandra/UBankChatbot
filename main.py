# -------------------------------------- Librerías
import os
from tkinter import *
from tkinter import messagebox

import json
import random
import re
from collections import Counter

import numpy as np
import pickle
import nltk
import tensorflow as tf
import tflearn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import codecs

stemmer = SnowballStemmer('spanish')

# Inicialización de variables
questions_asked = []
answers_given = []
preguntas_usuario = []
users = []
edades = []
generos = []
words = []
tags = []
docs_x = []
docs_y = []
parting_words = ['adios', 'adiós', 'hasta la proxima', 'hasta la próxima', 'chao', 'hasta luego', 'nos vemos',
                 'gracias']
tag_classify = ""

# --------------------------------------- Constucción chat
with open('content/preguntas-respuesta.json', encoding='utf-8') as file:
    data = json.load(file)

for content in data['contenido']:
    for pattern in content['posibles-preguntas']:
        # Tokeniza cada palabra en la frase
        wrds_list = nltk.word_tokenize(pattern)
        # Agrega las palabras tokenizadas a la lista de palabras
        words.extend(wrds_list)
        # Agrega la frase tokenizada a la lista de documentos
        docs_x.append(wrds_list)
        # Agrega el tag de la intención a la lista de etiquetas
        docs_y.append(content['tag'])

        if content['tag'] not in tags:  # Se almacenan todos los tags mientras no esten dentro
            tags.append(content['tag'])

# Eliminar duplicados y ordenar la lista de palabras
words = [stemmer.stem(w.lower()) for w in words if w != '?']  # Casteo de palabra
words = sorted(list(set(words)))
tags = sorted(tags)
print(tags)

# Crea una matriz de ceros con el número de etiquetas y palabras
training = []
output = []
out_empty = [0 for _ in range(len(tags))]

for x, doc in enumerate(docs_x):
    # Convertir cada palabra del documento en su raíz y crear un array de palabras
    bag = []
    wrds_list = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds_list:
            bag.append(1)
        else:
            bag.append(0)

    # Crear un array de salida con un 1 en la posición correspondiente a la etiqueta del documento
    output_row = out_empty[:]
    # Establece la posición correspondiente a la etiqueta en 1
    output_row[tags.index(docs_y[x])] = 1

    # Añadir los arrays de entrada y salida a los arrays de entrenamiento y salida
    training.append(bag)
    output.append(output_row)

# Convierte las listas de entrenamiento y salida en matrices numpy
training = np.array(training)
output = np.array(output)

# Guarda las palabras, etiquetas y entrenamiento en un archivo
with open('files/data.pickle', 'wb') as filePicker:
    pickle.dump((words, tags, training, output), filePicker)

# Carga las palabras, etiquetas y entrenamiento desde el archivo
with open('files/data.pickle', 'rb') as filePicker:
    words, tags, training, output = pickle.load(filePicker)

tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')  # Predicciones
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=10,
          show_metric=True)  # Va a ver la información 1000 veces y cuantas entradas
model.save('./files/modelo.tflearn')


# Función para clasificar una entrada y obtener la respuesta del chatbot
def classify_input(sentence):
    # Se llena de ceros una lista del mismo tamaño que la lista con las palabras tokenizadas
    bag = [0 for _ in range(len(words))]

    # Se crea una lista con las palabras que contiene la oración
    input_processed = nltk.word_tokenize(sentence)

    # Se pone en miniscula cada palabra y se reduce cada palabra a su raiz
    input_processed = [stemmer.stem(word.lower()) for word in input_processed]

    # Se compara la lista de palabras procesada y la lista de palabras anteriormente establecida
    for individual_word in input_processed:
        for i, word in enumerate(words):
            if word == individual_word:
                # Settea con 1 la posición equivalente en la lista bag en caso de coincidir
                bag[i] = 1

    # Se realiza la predicción según el modelo entrenado
    results = model.predict([np.array(bag)])
    index_results = np.argmax(results)
    tag = tags[index_results]
    tag_classify = tag

    for tagA in data['contenido']:
        if tagA['tag'] == tag:
            answers = tagA['posibles-respuesta']
            return tag_classify, random.choice(answers)

with open('content/base-datos.json', 'r', encoding='utf-8') as archivo:
    down_data = json.load(archivo)

def escribir_excel():
   
    categorias = [
    "saludo", "despedida", "opciones-principales", "cuenta-ahorros",
    "cuenta-ahorros-definicion", "cuenta-ahorros-obtenerla", "cuenta-ahorros-beenficios",
    "tarjeta-credito", "tarjeta-credito-definicion", "tarjeta-credito-beneficios",
    "tarjeta-credito-solicitud", "tarjeta-credito-tipos", "pagos", "pagos-definición", "pagos-tipos"]
    
    # Crear un diccionario para almacenar los datos
    datos = {'user': [], 'edad': [], 'genero': []} 
    for categoria in categorias:
        datos[categoria] = []

    # Procesar cada entrada del archivo JSON
    for entry in down_data['data']:
        usuario = entry['user']
        edad = entry['edad']
        genero = entry['genero']
        preguntas_usuario = entry['pregunta']
        datos['user'].append(usuario)
        datos['edad'].append(edad)
        datos['genero'].append(genero)
        # Comprobar si el usuario hizo preguntas en cada categoría
        # Crear un diccionario temporal para rastrear las categorías
        temp_data = {categoria: 0 for categoria in categorias}

        # Comprobar si el usuario hizo preguntas en cada categoría
        for pregunta in preguntas_usuario:
            tag, respuesta_chat = classify_input(pregunta)
            if tag in categorias:
                temp_data[tag] = 1

        # Agregar los valores al diccionario de datos
        for categoria in categorias:
            datos[categoria].append(temp_data[categoria])
    
    # Crear un DataFrame a partir del diccionario de datos
    df = pd.DataFrame(datos)
    df.to_excel('content/resultado.xlsx', index=False)

# ------------------------- Entrenamiento ------------------------------
with open('content/base-datos.json', 'r', encoding='utf-8') as archivo:
    collected_data = json.load(archivo)


# Se escribe el archivo JSON con todas las preguntas
def escribir_json():
    for i, user in enumerate(users):
        collected_data["data"].append({
            'user': user,
            'edad': edades[i],
            'genero': generos[i],
            'pregunta': questions_asked,
            'respuesta': [texto.strip().replace('\n            ', '') for texto in answers_given],
        })

    with open('content/base-datos.json', 'w', encoding='utf-8') as archivo:
        json.dump(collected_data, archivo, indent=4, ensure_ascii=False)


# Se obtienen todas las preguntas registradas en el archivo JSON
for user in collected_data["data"]:
    for pregunta in user["pregunta"]:
        preguntas_usuario.append(pregunta)


def explorar_datos():
    print('\r' + ' ' * 100 + '\r', end='')
    total_preguntas = len(preguntas_usuario)
    print('El total de preguntas de la exploración son:', total_preguntas, "\n")

    preguntas_frecuentes = Counter(preguntas_usuario).most_common(10)
    print('Las preguntas más frecuentes son: ', preguntas_frecuentes, "\n")

    if len(preguntas_usuario) > 0:
        pregunta_aleatoria = random.choice(preguntas_usuario)
        print('Una pregunta aleatoria de las que nos han proporcionado los usuarios es: ', pregunta_aleatoria, "\n")

    preguntas_procesadas = [[stemmer.stem(palabra) for palabra in word_tokenize(pregunta.lower())] for
                            pregunta in preguntas_usuario]
    palabras_preguntas = [palabra for pregunta in preguntas_procesadas for palabra in pregunta]
    palabras_frecuentes_preguntas = Counter(palabras_preguntas).most_common(10)
    print('Las palabras que se utilizan en las preguntas con su frecuencia son: ', palabras_frecuentes_preguntas, "\n")


# ------------------------------- Función principal ------------------------
def main(message):
   # explorar_datos()
    if message != '' and message != ' ':
        patron = r"mi nombre es (\w+)"
        patronEdad = r"mi edad es (\w+)"
        patronGenero = r"mi género es (\w+)"
        result = re.search(patron, message.lower())
        result01 = re.search(patronEdad, message.lower())
        result02 = re.search(patronGenero, message.lower())

        if result:
            users.append(result.group(1))
        
        if result01:
            edades.append(result01.group(1))

        if result02:
            generos.append(result02.group(1))
        

        # Entrada
        chatLog.insert(END, "_" * 38 + "\n\n    ")
        chatLog.image_create(END, image=imageUser)
        chatLog.insert(END, "  " + message + "\n")

        tag, respuesta_chat = classify_input(message)
        

        # Salida
        chatLog.insert(END, "_" * 38 + "\n\n    ")
        chatLog.image_create(END, image=imageChat)
        chatLog.insert(END, "  " + respuesta_chat + '\n')

        texto.set("")

        questions_asked.append(message)
        answers_given.append(respuesta_chat)

        if message.lower() in parting_words:
            escribir_json()
            root.destroy()

    else:
        messagebox.showerror("Error", "Por favor ingrese un texto valido")
    
    escribir_excel()


# ---------------------------------------  Interfaz gráfica ------------------------------------


# Inicialización de la ventana principal del chatBot
root = Tk()
root.geometry('400x600')
root.resizable(width=False, height=False)
root.configure(background="white")
lstChats = Listbox(root)

logoPic = PhotoImage(file='./images/mensaje.png')
imageChat = PhotoImage(file='./images/LogoUBank.png')
imageUser = PhotoImage(file='./images/Usuario.png')
send = PhotoImage(file='./images/send.png')
txt_image = PhotoImage(file='./images/textfield.png')

chatLog = Text(root, bd=0, bg="black", height="28.5", width="38", pady=10, padx=10)
scrollbar = Scrollbar(root, command=chatLog.yview, cursor="heart")
chatLog['yscrollcommand'] = scrollbar.set
chatLog.config(foreground="white", font=("Verdana", 11))
chatLog.place(x=0, y=0)

# Inicia el chat
chatLog.insert(END, "\n" + "_" * 38)
chatLog.insert(END, "\n\n    ")
chatLog.image_create(END, image=imageChat)
chatLog.insert(END, "   Hola, ¿cómo es tu nombre?\n")

Label(root, text="- UBank -", font=('Opensans', 14), bg="black", foreground="white").place(x=160, y=6)

# TEXTFIELD
texto = StringVar()
Label(root, image=txt_image, border=0, bg="white").place(x=32, y=553)
input_text = Entry(root, width=34, border=0, font=('bold', 11), bg="#EEEEEE", textvariable=texto)
input_text.place(x=42, y=560)

# BOTÓN ENVIAR
Button(root, image=send, command=lambda: main(texto.get()), background="white", borderwidth=0).place(x=335, y=556)


def on_keypress(event):
    if event.keysym == 'Return':
        main(texto.get())


input_text.bind('<KeyPress>', on_keypress)

root.title("UBank Chatbot")
root.iconphoto(False, logoPic)
root.mainloop()
