import logging
import os
import sys
from unittest.mock import MagicMock

# --- Hack: Simular torchvision roto para prevenir cuelgue de T5 al cargar ---
# El entorno del usuario tiene una instalación rota de torchvision (RuntimeError: operator torchvision::nms does not exist)
# T5 es solo texto, así que no necesitamos visión.
try:
    import torchvision
except (ImportError, RuntimeError):
    mock_tv = MagicMock()
    mock_tv.__spec__ = None # ¿Imitar un módulo que ha sido cargado? ¿O usar especificación explícita? 
    # Si se llama a find_spec, devuelve None si no se encuentra.
    # Pero el código del usuario podría comprobar __spec__.
    # En realidad, más simple: ¿anular registro de sys.modules para que importlib busque? No, queremos BLOQUEAR búsqueda válida.
    # Intentemos simplemente poner None y ver si importlib lo trata como 'no encontrado' o 'incorporado'.
    
    # Mejor: ¿Parchear find_spec? No. 
    # Intentemos ajustar __spec__ a uno ficticio.
    from importlib.machinery import ModuleSpec
    mock_tv.__spec__ = ModuleSpec(name="torchvision", loader=None)
    
    sys.modules["torchvision"] = mock_tv
    sys.modules["torchvision.transforms"] = MagicMock()
    sys.modules["torchvision.ops"] = MagicMock()

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configurar registro
logger = logging.getLogger("MangoManager")

class MangoManager:
    """
    Gestor para el modelo MANGO T5 (Sysadmin AI).
    Traduce lenguaje natural a comandos Bash.
    """
    def __init__(self, model_path=None):
        if model_path:
             self.model_path = model_path
        else:
             # Auto-detectar prioridad: Lime > MANGOt5
             if os.path.exists("models/Lime"):
                 self.model_path = "models/Lime"
             elif os.path.exists("models/MANGOt5"):
                 self.model_path = "models/MANGOt5"
             else:
                 self.model_path = "models/MANGOt5" # Respaldo por defecto
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.device = "cpu" # CPU por defecto para estabilidad en i3/8GB, puede cambiar a cuda si está disponible

        self.load_model()

    def load_model(self):
        """Carga el modelo y el tokenizer."""
        if not os.path.exists(self.model_path):
            logger.info(f"Directorio del modelo no encontrado: {self.model_path} (MANGO desactivado).")
            self.is_ready = False
            return

        try:
            logger.info(f"Cargando MANGO T5 desde {self.model_path}...")
            
            # Detectar dispositivo
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                # OPTIMIZACIÓN: Limitar hilos de PyTorch a 1 o 2 en CPUs dual-core (i3)
                # para evitar inanición de los hilos de audio/voz.
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                
            logger.info(f"Usando dispositivo: {self.device} (Optimized for Multi-tasking)")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            
            self.is_ready = True
            logger.info("MANGO T5 cargado correctamente.")
            
            # Limpieza de memoria
            import gc
            gc.collect()

        except Exception as e:
            logger.error(f"Error cargando MANGO T5: {e}", exc_info=True)
            self.is_ready = False

    def infer(self, text):
        """
        Genera un comando Bash a partir de texto.
        Retorna: (comando_str, confidence_score) o (None, 0)
        """
        if not self.is_ready or not text:
            return None, 0

        try:
            # Preprocesamiento simple
            input_text = text.strip()
            
            # Tokenizar
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generar
            outputs = self.model.generate(
                input_ids, 
                max_length=128, 
                num_beams=5, # Beam search para mejor calidad
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
            # Decodificar
            command = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Calcular confianza aproximada (heurística simple basada en puntuación de secuencia)
            # Las puntuaciones de logs de gen. de T5, pero por ahora confiamos en el primer beam.
            # Convertir log_score a prob aproximadamente: exp(score / length)
            sequence_score = outputs.sequences_scores[0].item()
            length = len(outputs.sequences[0])
            confidence = 0.0
            
            # --- Filtrado ---
            # Penalizar frases de chat conocidas o no-comandos muy cortos
            ignored_phrases = ["hola", "gracias", "entendido", "me he entendido", "buenos dias", "adios", "que tal"]
            if input_text.lower() in ignored_phrases or len(input_text.split()) < 2:
                 # A menos que sea un comando conocido de una sola palabra como "reboot" (que de todos modos necesita auth), ignorarlo
                 # Por seguridad, reducimos la confianza para estas entradas genéricas
                 confidence = 0.0
                 logger.info(f"Input '{input_text}' filtered as likely chat/noise.")
                 return None, 0.0

            # Normalizando algo arbitrariamente para T5 dado que los puntajes son probs logarítmicas negativas
            # Puntuaciones de T5 son habitualmente en torno de -1.0 a -8.0
            if sequence_score > -1.5: confidence = 0.98
            elif sequence_score > -3.0: confidence = 0.9
            elif sequence_score > -5.0: confidence = 0.75
            else: confidence = 0.5
            
            logger.info(f"Raw Score: {sequence_score}")

            logger.info(f"MANGO Input: '{text}' -> Output: '{command}' (Score: {sequence_score:.2f})")
            
            return command, confidence

        except Exception as e:
            logger.error(f"Error en inferencia MANGO: {e}")
            return None, 0
