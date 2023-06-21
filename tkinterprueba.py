import tkinter as tk

# Función que se ejecuta cuando el cursor se mueve
def on_mouse_move(event):
    # Coordenadas del cursor
    x = event.x
    y = event.y
    
    # Comprobar si el cursor está en un punto específico
    if (x, y) in puntos:
        # Dibujar un círculo alrededor del cursor
        canvas.create_oval(x - radio, y - radio, x + radio, y + radio, outline='red')

# Crear la ventana
window = tk.Tk()
window.title("Círculo alrededor del cursor")

# Crear un lienzo (canvas) en la ventana
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Definir los puntos específicos donde se mostrará el círculo
puntos = [(100, 100), (200, 300), (400, 200)]

# Definir el radio del círculo
radio = 10

# Asociar la función on_mouse_move al movimiento del cursor
canvas.bind('<Motion>', on_mouse_move)

# Ejecutar el bucle principal de la ventana
window.mainloop()