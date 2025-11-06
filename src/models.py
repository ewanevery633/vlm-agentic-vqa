import torch
import logging
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class QwenVLM:
    """
    A wrapper class for the Qwen2.5-VL-3B-Instruct model.
    Handles model loading, processing, and inference.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=None):
        try:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            logger.info(f"Loading model '{model_name}' to device '{self.device}'...")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else self.device
            )
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            logger.info("Model and processor loaded successfully.")

        except ImportError:
            logger.error("qwen-vl-utils not found. Please run 'pip install qwen-vl-utils==0.0.8'")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def inference(self, prompt="Describe this image.", image_path=None, max_new_tokens=128):
        """
        Performs inference with or without an image.
        """
        try:
            messages = [{"role": "user", "content": []}]
            image_inputs, video_inputs = None, None

            if image_path:
                try:
                    image = Image.open(image_path).convert("RGB")
                    messages[0]["content"].append({"type": "image", "image": image})
                    messages[0]["content"].append({"type": "text", "text": prompt})
                    image_inputs, video_inputs = process_vision_info(messages)
                except FileNotFoundError:
                    logger.error(f"Image file not found: {image_path}")
                    return [f"Error: Image file not found at {image_path}"]
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    return [f"Error: Could not process image. {e}"]
            else:
                messages[0]["content"].append({"type": "text", "text": prompt})

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return output_text

        except Exception as e:
            logger.error(f"Error during VLM inference: {e}", exc_info=True)
            return [f"Error: Inference failed. {e}"]
