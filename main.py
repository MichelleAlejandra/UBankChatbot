from tkinter import *
from tkinter import messagebox
import json

import numpy
import random
import tflearn
import tensorflow
import pickle
import nltk  # Para procesamiento de lenguaje natural
from nltk.stem.lancaster import LancasterStemmer  # Transformar palabras - quitar letras de más

stemmer = LancasterStemmer()

# Validación - Descarga paquete en caso de que moleste al ejecutarlo
nltk.download('punkt')

with open('content/content.json', encoding='utf-8') as archivo:
    datos = json.load(archivo)

try:
    with open('./archivos/variables.pickle', 'rb') as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except:
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos['contenido']:
        for patrones in contenido['patrones']:
            auxPalabra = nltk.word_tokenize(patrones)  # Toma la frase y la separa en palabras o tokens
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido['tag'])

            if contenido['tag'] not in tags:  # Se almacenan todos los tags mientras no esten dentro
                tags.append(contenido['tag'])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w != '?']  # Casteo de palabra
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = []
    salida = []
    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)

        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])] = 1  # Contenido de y que hay en indice y asigna un 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = numpy.array(entrenamiento)
    salida = numpy.array(salida)
    with open('./files/variables.pickle', 'wb') as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

tensorflow.compat.v1.reset_default_graph()

red_neuronal = tflearn.input_data(shape=[None, len(entrenamiento[0])])
red_neuronal = tflearn.fully_connected(red_neuronal, 10) #Cantidad de neuronas que queremos que tenga
red_neuronal = tflearn.fully_connected(red_neuronal, 10)
red_neuronal = tflearn.fully_connected(red_neuronal, len(salida[0]), activation='softmax')  # Predicciones
red_neuronal = tflearn.regression(red_neuronal)

modelo = tflearn.DNN(red_neuronal)
modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10,
               show_metric=True)  # Va a ver la información 1000 veces y 10 entradas (patrones)
modelo.save('./files/model.tflearn')
'''try:
    modelo.load('./files/model.tflearn')
except:
    modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10,
               show_metric=True)  # Va a ver la información 1000 veces y 10 entradas (patrones)
    modelo.save('./files/model.tflearn')'''


def main(message):
    if message != '' and message != ' ':
        # Entrada
        chatLog.insert(END,"_"*38 + "\n\n    ")
        chatLog.image_create(END, image=imageUser)
        chatLog.insert(END, "  " + message + "\n")

        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(message)
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in
                            entradaProcesada]


        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1

        resultados = modelo.predict([numpy.array(cubeta)])
        resultadosIndices = numpy.argmax(
            resultados)
        tag = tags[resultadosIndices]

        for tagAux in datos['contenido']:
            if tagAux['tag'] == tag:
                respuesta = tagAux['respuestas']

        respuestaChat = " " + random.choice(respuesta)  # Respuesta escogida al azar dentro de las posibles respuestas

        # Salida
        chatLog.insert(END, "_" * 38 + "\n\n    ")
        chatLog.image_create(END, image=imageChat)
        chatLog.insert(END, "  " + respuestaChat + '\n')

        texto.set("")
    else:
        messagebox.showerror("Error", "Por favor ingrese un texto valido")

# Interfaz grafica
#Inicialización de la ventana principal del chatBot
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
chatLog.insert(END,  "\n")
'''chatLog.insert(END,  "\n" + "_"*48)
chatLog.insert(END, "\n\n    ")
chatLog.image_create(END, image=imageChat)
chatLog.insert(END, "   Hola, ¿como puedo ayudarte?\n")'''

Label(root, text="- UBank -", font=('Opensans', 14), bg="black", foreground="white").place(x=160, y=6)

# TEXTFIELD
texto = StringVar()
eImage = Label(root,image=txt_image, border= 0, bg="white").place(x=32, y=553)
txt = Entry(root, width=34, border=0, font=('bold', 11), bg="#EEEEEE", textvariable=texto).place(x=42, y=560)

# BOTÓN ENVIAR
Button(root, image=send, command=lambda:main(texto.get()), background="white", borderwidth=0).place(x=335, y=556)

root.title("UBank Chatbot")
root.iconphoto(False, logoPic)
root.mainloop()