# -*- coding: utf-8 -*-
"""
LABORATORIO 2: CONVOLUCIÓN 2D

Autor: Matias Hernandez
Ayudante: Miguel Salinas
Profesor: Alejandro Catalan  
"""

'''
Importacion de Librerias
'''
import numpy as np
import matplotlib.image as mpi
import matplotlib.pyplot as mpp

'''
Funciones
'''
# Entrada: nombre (String de la ruta de la imagen)
# Salida: imagen (Arreglo con la imagen normalizada)
# Objetivo: Leer el archivo "nombre" y retornar la imagen normalizada en escala de grises

def readImagen(nombre):
    imagen = mpi.imread(nombre)
    return imagen / 255 

# Entradas:
#   imagen (Arreglo con la imagen)
#   nombre (String de la ruta de la imagen)
#   titulo (string con el nombre de la imagen)
#   filtro (booleano, true: la imagen es filtro, false: la imagen es espectro)
# Objetivo: Guardar la imagen en la ruta "nombre", aplica cmap gris si filtro es true

def saveImagen(imagen, nombre, titulo, filtro):
    if filtro:
        mpp.imshow(imagen, cmap="gray",vmin=0, vmax=1)
    else:
        mpp.imshow(imagen)
    mpp.title(titulo)
    imagen = mpp.gcf()
    imagen.savefig(nombre, dpi=100)
    
# Entradas: 
#   imagen (Arreglo con la imagen)
#   titulo (string con el nombre de la imagen)
#   filtro (booleano, true: la imagen es filtro, false: la imagen es espectro)
# Objetivo: Mostrar la imagen en la ruta "nombre", aplica cmap gris si filtro es true

def verImagen(imagen, titulo, filtro):
    if filtro:
        mpp.imshow(imagen, cmap="gray",vmin=0, vmax=1)
    else:
        mpp.imshow(imagen)
    mpp.title(titulo)
    mpp.show()

# Entradas :
#   imagen (Arreglo con la imagen)
#   kernel (Arreglo con el filtro) 
# Salida: matrizContorno
# Objetivo: Copia la imagen original en otra matriz rodeada de un contorno de ceros del tamaño necesario para aplicar la convolución.

def matrizContornoCero(imagen, kernel):
    FilasKernel, ColuKernel = kernel.shape
    FilasImg, ColuImg = imagen.shape
    PosFilaMatrizCeros = FilasKernel - 1
    PosColuMatrizCeros = ColuKernel - 1 
    PosFilaImg = 0
    PosColuImg = 0

    matrizContorno = np.zeros((FilasImg + 2 * PosFilaMatrizCeros, ColuImg + 2 * PosColuMatrizCeros))

    for i in range(PosFilaMatrizCeros, PosFilaMatrizCeros + FilasImg):
        for j in range(PosColuMatrizCeros, PosColuMatrizCeros + ColuImg):
            matrizContorno[i][j] = imagen[PosFilaImg][PosColuImg]
            PosColuImg+=1
        PosFilaImg += 1
        PosColuImg = 0
    
    return matrizContorno

# Entradas :
#   imagen (Arreglo con la imagen)
#   kernel (Arreglo con el filtro) 
# Salida: convolucion (Matriz con imagen despues de la convolucion)
# Objetivo: Aplica convolución en 2 dimensiones entre imagen y kernel.

def convolucion2D(imagen, kernel):
    convolucion = []

    # Matriz
    matrizContorno = matrizContornoCero(imagen, kernel)
    FilasMC, ColuMC = matrizContorno.shape

    # Kernel
    kernel = np.flipud(np.fliplr(kernel))
    FilasKernel, ColuImg = kernel.shape

    # Iteradores
    iterFilas = FilasMC - FilasKernel + 1
    iterColumnas = ColuMC - ColuImg + 1
    PosFilaConvolucion = 0

    for fila in range(iterFilas):
        # Se inicia la fila 
        convolucion.append([]) 
        for col in range(iterColumnas):
            # Se reinicia la cuenta en cada columna
            suma = 0 
            for filasK in range(FilasKernel):
                for columnI in range(ColuImg):
                    suma += kernel[filasK][columnI] * matrizContorno[fila + filasK][col + columnI]
            convolucion[PosFilaConvolucion].append(suma)
        PosFilaConvolucion += 1
    return convolucion

# Entrada: imagen (Arreglo con la imagen)
# Salida: Matriz del espectro de Fourier (Datos de la transformada de Fourier de la imagen)
# Objetivo: Calcula la transformada de Fourier en 2D para una imagen dada.

def fourier(imagen):
    np.seterr(divide = 'ignore') #Configuración necesaria (recomendacion de internet)
    # Transformada de Fourier 
    transformadaFourier = np.fft.fft2(imagen)

    # Shift de zero a la Transformada 
    transformadaFourierShift = np.fft.fftshift(transformadaFourier)

    # Retorno del valor normalizado positivo 
    return 20 * np.log(np.abs(transformadaFourierShift))


'''
BLOQUE PRINCIPAL
'''

filtroBordes = np.array([[1,2,0,-2,-1],
                         [1,2,0,-2,-1],
                         [1,2,0,-2,-1],
                         [1,2,0,-2,-1],
                         [1,2,0,-2,-1]]).astype('float') # Kernel Bordes
filtroGauss = np.array([[1,4,6,4,1],
                            [4,16,24,16,4],
                            [6,24,36,24,6],
                            [4,16,24,16,4],
                            [1,4,6,4,1]]).astype('float') * (1/256) # Kernel Gauss

'''
Pregunta 2
'''
filtroOutline = np.array([[-1,-1,-1],
                         [-1,8,-1],
                         [-1,-1,-1]]).astype('float') # Kernel outline


pregunta2 = readImagen('lena512.bmp')
verImagen(pregunta2,'Imagen pregunta 2',True)
print('Convolucion 2D con kernel Outline')
pregunta2Conv = convolucion2D(pregunta2, filtroOutline)
verImagen(pregunta2Conv,'Test Convolucion 2D con kernel Outline',True)

'''
Pregunta 5
'''
imagen = readImagen('lena512.bmp')
verImagen(imagen,'Imagen Original',True)

# Aplicacion de filtro de suavizado gaussiano
imagenFG = convolucion2D(imagen, filtroGauss)

# Aplicacion de filtro detector de bordes
imagenFB = convolucion2D(imagen, filtroBordes)

# Mapeo de imagen filtrada suavizado Gaussiano
verImagen(imagenFG,'Imagen Suavizado Gaussiano',True)
saveImagen(imagenFG,'./SuavizadaGauss.png','Imagen Suavizada Gaussiano',True)

# Mapeo de imagen filtrada bordes
verImagen(imagenFB,'Imagen Detector Bordes',True)
saveImagen(imagenFB,'./DetectorBordes.png','Imagen Detector Bordes',True)

'''
Pregunta 6
'''

# Aplicacion transformada de fourier a la imagen 
espectroImg = fourier(imagen)

# Mapeo fourier a la imagen 
verImagen(espectroImg,'Espectro Fourier Imagen Original',False)
saveImagen(espectroImg,'./Fourier_Original.png','Espectro Fourier Imagen Original',False)

## Aplicacion transformada de fourier imagen filtrada suavizado ##
espectroImgFG = fourier(imagenFG)

## Mapeo fourier imagen filtrada suavizado ##
verImagen(espectroImgFG,'Espectro Fourier Imagen Suavizada con Gauss',False)
saveImagen(espectroImgFG,'./Fourier_SuavizadoGauss.png','Espectro Fourier Imagen Suavizada con Gauss',False)

## Aplicacion transformada de fourier imagen filtrada bordes ##
espectroImgFB = fourier(imagenFB)

## Mapeo fourier imagen filtrada bordes ##
verImagen(espectroImgFB,'Espectro Fourier Imagen Bordes Detectados',False)
saveImagen(espectroImgFB,'./Fourier_Bordes.png','Espectro Fourier Imagen Bordes Detectados',False)






