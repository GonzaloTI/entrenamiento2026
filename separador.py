import os
import random
import shutil

# ==========================================
# 1. CONFIGURA TUS RUTAS AQUÍ
# ==========================================
# Carpeta original donde están todas tus fotos mezcladas
ruta_imagenes = "C:/Users/Usuario/Documents/python/yolo/entrenamiento/2026/yolo/train/images" 
# Carpeta original donde están todos tus .txt mezclados
ruta_etiquetas = "C:/Users/Usuario/Documents/python/yolo/entrenamiento/2026/yolo/train/labels"     
# Carpeta principal donde el script creará la estructura para YOLO
ruta_salida = "C:/Users/Usuario/Documents/python/yolo/entrenamiento/2026/yolosalida"

# ==========================================
# 2. PORCENTAJES (80% Train, 10% Val, 10% Test)
# ==========================================
train_ratio = 0.80
val_ratio = 0.10
# El test será automáticamente el 10% restante

def crear_carpetas():
    carpetas = ['train', 'valid', 'test']
    subcarpetas = ['images', 'labels']
    for c in carpetas:
        for s in subcarpetas:
            os.makedirs(os.path.join(ruta_salida, c, s), exist_ok=True)

def dividir_dataset():
    # Obtener todas las imágenes (filtra por extensiones comunes)
    archivos_img = [f for f in os.listdir(ruta_imagenes) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Mezclar las imágenes aleatoriamente para evitar sesgos
    random.seed(42)
    random.shuffle(archivos_img)
    
    # Calcular cantidades exactas
    total = len(archivos_img)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    # Dividir las listas
    train_imgs = archivos_img[:train_count]
    val_imgs = archivos_img[train_count:train_count + val_count]
    test_imgs = archivos_img[train_count + val_count:]
    
    print(f"Total encontradas: {total} imágenes.")
    print(f"Repartiendo: Train({len(train_imgs)}) | Val({len(val_imgs)}) | Test({len(test_imgs)})")

    # Función interna para mover los archivos
    def copiar_archivos(lista_archivos, carpeta_destino):
        for img_name in lista_archivos:
            # Rutas de origen
            origen_img = os.path.join(ruta_imagenes, img_name)
            
            # El archivo txt tiene el mismo nombre que la imagen
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            origen_txt = os.path.join(ruta_etiquetas, txt_name)
            
            # Rutas de destino (YOLO format: images/ y labels/)
            destino_img = os.path.join(ruta_salida, carpeta_destino, 'images', img_name)
            destino_txt = os.path.join(ruta_salida, carpeta_destino, 'labels', txt_name)
            
            # Copiar imagen
            shutil.copy(origen_img, destino_img)
            
            # Copiar etiqueta (solo si existe, previene errores)
            if os.path.exists(origen_txt):
                shutil.copy(origen_txt, destino_txt)
            else:
                print(f"Advertencia: Falta el archivo .txt para {img_name}")

    print("Copiando a Train...")
    copiar_archivos(train_imgs, 'train')
    
    print("Copiando a Val...")
    copiar_archivos(val_imgs, 'val')
    
    print("Copiando a Test...")
    copiar_archivos(test_imgs, 'test')
    
    print("\n¡Proceso finalizado con éxito! Revisa tu carpeta de salida.")

if __name__ == '__main__':
    crear_carpetas()
    dividir_dataset()