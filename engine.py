import logging
import os
import sys
from unittest.mock import MagicMock

# --- Hack: Mock broken torchvision to prevent T5 load crash ---
# The user's environment has a broken torchvision install (RuntimeError: operator torchvision::nms does not exist)
# T5 is text-only, so we don't need vision.
try:
    import torchvision
except (ImportError, RuntimeError):
    mock_tv = MagicMock()
    mock_tv.__spec__ = None # Imitate a module that has been loaded? Or use explicit spec? 
    # If find_spec is called, it returns None if not found.
    # But user code might check __spec__.
    # Actually, simpler: unregister it from sys.modules so importlib searches? No, we want to BLOCK valid search.
    # Let's try to just set None and see if importlib treats it as 'not found' or 'built-in'.
    
    # Better: Patch find_spec? No. 
    # Let's try to set __spec__ to a dummy.
    from importlib.machinery import ModuleSpec
    mock_tv.__spec__ = ModuleSpec(name="torchvision", loader=None)
    
    sys.modules["torchvision"] = mock_tv
    sys.modules["torchvision.transforms"] = MagicMock()
    sys.modules["torchvision.ops"] = MagicMock()

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup Logging
logger = logging.getLogger("MangoManager")

class MangoManager:
    """
    Gestor para el modelo MANGO T5 (Sysadmin AI).
    Traduce lenguaje natural a comandos Bash.
    """
    def __init__(self, model_path="models/MANGOt5"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.is_ready = False
        self.device = "cpu" # Default to CPU for stability on i3/8GB, can change to cuda if available

        self.load_model()

    def load_model(self):
        """Carga el modelo y el tokenizer."""
        if not os.path.exists(self.model_path):
            logger.error(f"Directorio del modelo no encontrado: {self.model_path}")
            return

        try:
            logger.info(f"Cargando MANGO T5 desde {self.model_path}...")
            
            # Detect device
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                # OPTIMIZATION: Limit PyTorch threads to 1 or 2 on dual-core CPUs (i3)
                # to prevent starving the audio/voice threads.
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                
            logger.info(f"Usando dispositivo: {self.device} (Optimized for Multi-tasking)")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(self.device)
            
            self.is_ready = True
            logger.info("MANGO T5 cargado correctamente.")
            
            # Memory Cleanup
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
            # Preprocessing simple
            input_text = text.strip()
            
            # Tokenize
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate
            outputs = self.model.generate(
                input_ids, 
                max_length=128, 
                num_beams=5, # Beam search para mejor calidad
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
            # Decode
            command = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Calcular confianza aproximada (simple heuristic based on sequence score)
            # T5 gen logs scores, but for now we trust the top beam.
            # Convert log_score to prob roughly: exp(score / length)
            sequence_score = outputs.sequences_scores[0].item()
            length = len(outputs.sequences[0])
            confidence = 0.0
            
            # --- Filtering ---
            # Penalize known chat phrases or very short non-commands
            ignored_phrases = ["hola", "gracias", "entendido", "me he entendido", "buenos dias", "adios", "que tal"]
            if input_text.lower() in ignored_phrases or len(input_text.split()) < 2:
                 # Unless it's a known single-word command like "reboot" (which usually needs auth anyway), ignore it
                 # For safety, we drop confidence for these generic inputs
                 confidence = 0.0
                 logger.info(f"Input '{input_text}' filtered as likely chat/noise.")
                 return None, 0.0

            # Normalizing somewhat arbitrarily for T5 since scores are negative log probs
            # T5 scores are usually around -1.0 to -8.0
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
