import os
import sys
import platform
import torch
from ultralytics import YOLO

def export_model(
    model_path="best_fragaria_5nu.pt",
    imgsz=640,
    export_format="onnx",  # 'onnx' o 'engine'
    half=False,            # Si es ONNX, suele ser mejor dejarlo en FP32 y hacer --fp16 en trtexec
    opset=12,
    device=None
):
    if not os.path.exists(model_path):
        print(f"[ERROR] El archivo de pesos '{model_path}' no existe en este directorio.")
        return

    # Auto-seleccionar dispositivo (GPU si está disponible)
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("🚀 EXPORTADOR DE MODELOS YOLO PARA NVIDIA JETSON NANO")
    print("=" * 65)
    print(f"📁 Modelo origen:      {model_path}")
    print(f"📐 Tamaño de imagen:   {imgsz}x{imgsz}")
    print(f"⚡ Formato objetivo:   {export_format.upper()}")
    print(f"🖥️  Dispositivo:        {device} ({'GPU CUDA' if device == 0 else 'CPU'})")
    print(f"🎯 Modo FP16 (Half):   {half}")
    print(f"⚙️  ONNX Opset:         {opset}")
    print("=" * 65)

    machine = platform.machine().lower()
    is_jetson = "aarch64" in machine or "arm" in machine

    if export_format == "engine" and not is_jetson:
        print("\n⚠️  ¡ADVERTENCIA IMPORTANTE SOBRE MOTORES TENSORRT (.ENGINE)! ⚠️")
        print("Un archivo .engine compilado en esta PC NO será compatible con")
        print("la Jetson Nano (arquitectura ARM64 / Maxwell sm_53).")
        print("Los archivos .engine son binarios específicos para cada chip.")
        print("\n👉 Para usarlo en Jetson Nano:")
        print("   1. Exporta a .ONNX aquí en tu PC.")
        print("   2. Pasa el archivo .onnx a tu Jetson Nano.")
        print("   3. Ejecuta el comando 'trtexec' directamente en la Jetson.\n")
        print("=" * 65 + "\n")

    print("[1/2] Cargando modelo...")
    model = YOLO(model_path)

    print(f"[2/2] Exportando modelo a '{export_format}'...")
    try:
        # Argumentos de exportacion
        export_args = {
            "format": export_format,
            "imgsz": imgsz,
            "dynamic": False,      # Dimensiones fijas: esencial para máxima velocidad en TensorRT
            "simplify": True,      # Optimiza grafos y fusiona capas
            "device": device
        }

        if export_format == "onnx":
            export_args["opset"] = opset
            if half and device != "cpu":
                export_args["half"] = True
        elif export_format == "engine":
            export_args["half"] = half

        exported_path = model.export(**export_args)

        # Si exportamos a ONNX, asegurar que sea un único archivo autocontenido
        if export_format == "onnx":
            try:
                import onnx
                onnx_model = onnx.load(str(exported_path))
                onnx.save(onnx_model, str(exported_path), save_as_external_data=False)
                data_extra = str(exported_path) + ".data"
                if os.path.exists(data_extra):
                    os.remove(data_extra)
            except Exception:
                pass

        print("\n" + "=" * 65)
        print("✅ ¡Exportación completada exitosamente!")
        print(f"📦 Archivo generado: {exported_path}")
        print("=" * 65)

        base_name = os.path.splitext(os.path.basename(model_path))[0]
        onnx_file = f"{base_name}.onnx"

        print("\n📋 PASOS A SEGUIR EN LA JETSON NANO (2GB RAM):")
        print(f"1. Transfiere '{onnx_file}' a la Jetson Nano (mediante USB, FileZilla o SCP).")
        print("2. Abre la terminal en tu Jetson Nano y corre:")
        print("-----------------------------------------------------------------")
        print(f"/usr/src/tensorrt/bin/trtexec --onnx={onnx_file} \\")
        print(f"    --saveEngine={base_name}.engine \\")
        print(f"    --fp16 \\")
        print(f"    --workspace=1024")
        print("-----------------------------------------------------------------")
        print("💡 Explicación de los parámetros:")
        print("   * --fp16: Activa precisión de 16 bits (duplica la velocidad en la Jetson).")
        print("   * --workspace=1024: Limita la memoria de compilación a 1GB para evitar")
        print("                       que la Jetson de 2GB sufra un reinicio por Out-Of-Memory.")
        print("=" * 65)

    except Exception as e:
        print(f"\n❌ Error durante la exportación: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exportar modelo YOLO a ONNX / TensorRT Engine para Jetson Nano")
    parser.add_argument("--weights", type=str, default="best_fragaria2.pt", help="Ruta al archivo .pt")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"], help="Formato de salida (onnx o engine)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolución de entrada (ej: 640, 416, 320)")
    parser.add_argument("--opset", type=int, default=12, help="Versión ONNX Opset (12 recomendado para TensorRT 8)")
    parser.add_argument("--half", action="store_true", help="Exportar pesos directamente en FP16")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo (0, 'cpu', etc.)")

    args = parser.parse_args()
    export_model(
        model_path=args.weights,
        imgsz=args.imgsz,
        export_format=args.format,
        half=args.half,
        opset=args.opset,
        device=int(args.device) if args.device and args.device.isdigit() else args.device
    )
