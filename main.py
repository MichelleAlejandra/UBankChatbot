# -------------------------------------- Librerías

from tkinter import *
from tkinter import messagebox
import json
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer #Transformar palabras - quitar letras demas
stemmer = LancasterStemmer()
import numpy as np
import tensorflow as tf
import tflearn
import random
import re
from collections import Counter

# Inicialización de variables
u_quest = []
u_ans = []
quetions_with_user = []
preguntas_usuario = []
users = []
words = []
tags = []
docs_x = []
docs_y = []


# --------------------------------------- Entrenamiento chat
with open('./preguntas-respuesta.json', encoding='utf-8') as file:
    data = json.load(file)

for content in data['contenido']:
    for pattern in content['posibles-preguntas']:
        # Tokeniza cada palabra en la frase
        wrds_list  = nltk.word_tokenize(pattern) 
        # Agrega las palabras tokenizadas a la lista de palabras
        words.extend(wrds_list)
        # Agrega la frase tokenizada a la lista de documentos
        docs_x.append(wrds_list)
        # Agrega el tag de la intención a la lista de etiquetas
        docs_y.append(content['tag']) 

        if content['tag'] not in tags: #Se almacenan todos los tags mientras no esten dentro
            tags.append(content['tag'])

# Eliminar duplicados y ordenar la lista de palabras
words = [stemmer.stem(w.lower()) for w in words if w!='?'] #Casteo de palabra
words = sorted(list(set(words)))
tags = sorted(tags)

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
    
    #Añadir los arrays de entrada y salida a los arrays de entrenamiento y salida
    training.append(bag)
    output.append(output_row)

# Convierte las listas de entrenamiento y salida en matrices numpy
training = np.array(training)
output = np.array(output)

# Guarda las palabras, etiquetas y entrenamiento en un archivo
with open('data.pickle','wb') as fileP:
    pickle.dump((words, tags, training, output), fileP)

# Carga las palabras, etiquetas y entrenamiento desde el archivo
with open('data.pickle', 'rb') as fileP:
    words, tags, training, output = pickle.load(fileP)

tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,10)
net = tflearn.fully_connected(net,10)
net = tflearn.fully_connected(net,len(output[0]),activation='softmax') # Predicciones
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch= 1000, batch_size=10, show_metric=True) #Va a ver la información 1000 veces y cuantas entradas
model.save('./files/modelo.tflearn')

# Función para clasificar una entrada y obtener la respuesta del chatbot
def classify(sentence):
   
    bag = [0 for _ in range(len(words))]
    input_processed = nltk.word_tokenize(sentence)
    input_processed = [stemmer.stem(word.lower()) for word in input_processed]
    
    for individual_word in input_processed:
        for i, word in enumerate(words):

            if word == individual_word:
                bag[i] = 1
    
    results = model.predict([np.array(bag)])
    index_results = np.argmax(results)
    tag = tags[index_results]

    for tagA in data['contenido']:
        if tagA['tag'] == tag:
            answers = tagA['posibles-respuesta']
            return random.choice(answers)

# ------------------------------------------------------- Entrenamiento
with open('base-datos.json', 'r') as archivo:
    datos = json.load(archivo)

# Se escribe el archivo JSON con todas las preguntas
def escribirJson():
    data = []
    for user in users:
        quetions_with_user.append(datos)
        quetions_with_user.append({
            'user': user,   
            'pregunta' : u_quest, 
            'respuesta' : u_ans, 
        })

    with open('base-datos.json', 'w') as archivo:
        json.dump(quetions_with_user,archivo,indent=4, ensure_ascii=False)


# Se obtienen todas las preguntas registradas en el archivo JSON
for user in datos["data"]:
    for pregunta in user["pregunta"]:
        preguntas_usuario.append(pregunta)

def explorarDatos():

    total_preguntas = len(preguntas_usuario)
    print('El total de preguntas de la exploración son:', total_preguntas, "\n")

    preguntas_frecuentes = Counter(preguntas_usuario).most_common(10)
    print('Las preguntas más frecuentes son: ', preguntas_frecuentes, "\n")

    pregunta_aleatoria = random.choice(preguntas_usuario)
    print('Una pregunta aleatoria de las que nos han proporcionado los usuarios es: ',pregunta_aleatoria, "\n")
    
    preguntas_procesadas = [[LancasterStemmer().stem(palabra) for palabra in word_tokenize(pregunta.lower())] for pregunta in preguntas_usuario]
    palabras_preguntas = [palabra for pregunta in preguntas_procesadas for palabra in pregunta]
    palabras_frecuentes_preguntas = Counter(palabras_preguntas).most_common(10)
    print('Las palabras que se utilizan en las preguntas con su frecuencia son: ' ,palabras_frecuentes_preguntas, "\n")



# ------------------------------------------------------- Función principal
def main(message):
    
    explorarDatos()

    if message != '' and message != ' ':
        
        patron = r"mi nombre es (\w+)"
        result = re.search(patron, message)
        if result:
            users.append(result.group(1))

        # Entrada
        chatLog.insert(END,"_"*48 + "\n\n    ")
        chatLog.image_create(END, image=imageUser)
        chatLog.insert(END, "  " + message + "\n")

        answer = classify(message)

        # Salida
        chatLog.insert(END, "_" * 48 + "\n\n    ")
        chatLog.image_create(END, image=imageChat)
        chatLog.insert(END, "  " + answer + '\n')

        texto.set("")

        u_quest.append(message)
        u_ans.append(answer)

        if message == 'Adios' or message == 'Hasta la proxima' or message == 'Chao' or message == 'Hasta luego' or message == 'Hasta pronto' or message == 'Nos vemos' or message == 'Gracias' :
            escribirJson()
            
            root.destroy()

    else:
        messagebox.showerror("Error", "Por favor ingrese un texto valido")

#-------------------------------------------------------------- Interfaz grafica
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

chatLog = Text(root, bd=0, bg="black", height="28.5", width="50", pady=10, )
scrollbar = Scrollbar(root, command=chatLog.yview, cursor="heart")
chatLog['yscrollcommand'] = scrollbar.set
chatLog.config(foreground="white", font=("Verdana", 11))
chatLog.place(x=0, y=0)

# Inicia el chat
chatLog.insert(END,  "\n" + "_"*48)
chatLog.insert(END, "\n\n    ")
chatLog.image_create(END, image=imageChat)
chatLog.insert(END, "   Hola, ¿cómo es tu nombre?\n")


Label(root, text="- UBank -", font=('Opensans', 14), bg="black", foreground="white").place(x=160, y=6)

# TEXTFIELD
texto = StringVar()
eImage = Label(root,image=txt_image, border= 0, bg="white").place(x=32, y=553)
txt = Entry(root, width=34, border=0, font=('bold', 11), bg="#EEEEEE", textvariable=texto).place(x=42, y=560)

# BOTÓN ENVIAR
Button(root, image=send, command=lambda:main(texto.get()), background="white", borderwidth=0).place(x=335, y=556)

root.title("Chatbot Consultor de Mascotas")
root.iconphoto(False, logoPic)
root.mainloop()



