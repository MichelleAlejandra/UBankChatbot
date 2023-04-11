from tkinter import *
from tkinter import messagebox


def main(message):
    if message != '' and message != ' ':
        # Entrada
        chatLog.insert(END,"_"*48 + "\n\n    ")
        chatLog.image_create(END, image=imageUser)
        chatLog.insert(END, "  " + message + "\n")

        # Salida
        chatLog.insert(END, "_" * 48 + "\n\n    ")
        chatLog.image_create(END, image=imageChat)
        chatLog.insert(END, "  " + message + '\n')

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

chatLog = Text(root, bd=0, bg="black", height="28.5", width="50", pady=10, )
scrollbar = Scrollbar(root, command=chatLog.yview, cursor="heart")
chatLog['yscrollcommand'] = scrollbar.set
chatLog.config(foreground="white", font=("Verdana", 11))
chatLog.place(x=0, y=0)

# Inicia el chat
chatLog.insert(END,  "\n" + "_"*48)
chatLog.insert(END, "\n\n    ")
chatLog.image_create(END, image=imageChat)
chatLog.insert(END, "   Hola, ¿como puedo ayudarte?\n")


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