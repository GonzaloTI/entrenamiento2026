# Dataset: Detección de Enfermedades en Frutillas

**Origen:** Roboflow Universe (`hanoi-university-of-industry-xfhdu / acv-strawberry-disease`)  
**Tarea de IA:** Segmentación de instancias (*Instance Segmentation*)  
**Licencia:** CC BY 4.0  
**Número de Clases:** 5  

---

## Tabla de Clasificación de Etiquetas

| Etiqueta (Vietnamita) | Nombre técnico / Nombre común | Qué identifica en la fruta / planta |
| :--- | :--- | :--- |
| **`Benh cao su`** | **Podredumbre de cuero** *(Leather Rot / Phytophthora cactorum)* | En vietnamita agrícola se traduce como "enfermedad del caucho". La fruta se vuelve dura, seca, correosa o con textura gomosa, adquiriendo tonos marrón/pálido. |
| **`Benh dom den`** | **Antracnosis / Mancha negra** *(Black spot / Colletotrichum acutatum)* | Lesiones circulares oscuras, sutilmente hundidas y necróticas en la superficie del fruto o tallos. |
| **`Benh moc xam`** | **Moho gris** *(Gray mold / Botrytis cinerea)* | Capa o micelio de color grisáceo que recubre la fruta, muy frecuente en entornos de alta humedad. |
| **`Benh phan trang`** | **Oídio / Cenicilla** *(Powdery mildew)* | Capa o polvillo blanquecino/ceniciento característico sobre la fruta o la superficie foliar. |
| **`Qua binh thuong`** | **Fruta normal / sana** | Frutillas en buen estado, sin signos visibles de patologías o manchas. |

---

## Estructura de Configuración (YOLO)

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 5
names: ['Benh cao su', 'Benh dom den', 'Benh moc xam', 'Benh phan trang', 'Qua binh thuong']
```