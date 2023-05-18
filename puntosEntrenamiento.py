import matplotlib.pyplot as plt

# Tuplas anteriores
rectangulo_grande = [(0, 0), (800, 0), (1600, 0), (0, 450),(800, 450),(1600, 450),(0, 900),(800, 900),(1600, 900)]

# Coordenadas de los rectángulos pequeños (dentro del grande)
rectangulo_pequeno_1 = [(100, 50),(800, 50),(1500, 50),(100, 450),(1500, 450),(100, 850),(800, 850),(1500, 850)]
rectangulo_pequeno_2 = [(200, 100),(800, 100),(1400, 100),(200, 450),(1400, 450),(200, 800),(800, 800),(1400, 800)]
rectangulo_pequeno_3 = [(300, 150),(800, 150),(1300, 150),(300, 450),(1300, 450),(300, 750),(800, 750),(1300, 750)]
rectangulo_pequeno_4 = [(400, 200),(800, 200),(1200, 200),(400, 450),(1200, 450),(400, 700),(800, 700),(1200, 700)]
rectangulo_pequeno_5 = [(500, 250),(800, 250),(1100, 250),(500, 450),(1100, 450),(500, 650),(800, 650),(1100, 650)]
rectangulo_pequeno_6 = [(600, 300),(600, 600),(1000, 600),(1000, 300),(600, 450),(1000, 450),(800, 300),(800, 600)]
sector_izquierda = [(100,300 ),(250, 300),(400, 300),(100,600),(250, 600),(400, 600)]
sector_derecha = [(1500,300 ),(1350, 300),(1200, 300),(1500,600),(1350, 600),(1200, 600)]
sector_izquierda2 = [(600,200 ),(600, 125),(600, 50),(600,700),(600, 775),(600, 850)]
sector_derecha2 = [(1000,200 ),(1000, 125),(1000, 50),(1000,700),(1000, 775),(1000, 850)]



# Combinar ambas listas de tuplas
vector_tuplas = rectangulo_grande + rectangulo_pequeno_1 + rectangulo_pequeno_2 +rectangulo_pequeno_3 + rectangulo_pequeno_4+ rectangulo_pequeno_5 + sector_izquierda +sector_derecha +sector_izquierda2 + sector_derecha2 + rectangulo_pequeno_6

# Extraer las coordenadas x e y por separado
x_coords = [tupla[0] for tupla in vector_tuplas]
y_coords = [tupla[1] for tupla in vector_tuplas]

# Graficar las tuplas en rojo
plt.scatter(x_coords, y_coords, color='red')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.title('Gráfico de tuplas')
plt.grid(True)
plt.show()