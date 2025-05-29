import torch
import re
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
import numpy as np
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(SCRIPT_DIR, "../molmoe-1b")
MODEL = os.path.abspath(MODEL)
import pdb


class MolmoModel:
    def __init__(self, model_path=MODEL):
    # def __init__(self, model_path="allenai/Molmo-7B-O-0924"):
        """
        Load the Molmo model and processor from the specified path.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()
        print("✅ Model and processor loaded successfully!")

    def generate_response_from_image(self, image_input, text_prompt: str) -> str:
        """
        Generate a response using a raw image (np.ndarray or PIL.Image) and a text prompt.
        """
        from PIL import Image, ImageStat
        import torch
        from torch import autocast

        # Convert input to PIL.Image
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise TypeError("Expected image_input to be np.ndarray or PIL.Image.Image")

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to save memory
        image = image.resize((224, 224))

        # Convert transparent images (Molmo doesn’t like them)
        if image.mode == "RGBA":
            gray_image = image.convert('L')
            stat = ImageStat.Stat(gray_image)
            average_brightness = stat.mean[0]
            bg_color = (0, 0, 0) if average_brightness > 127 else (255, 255, 255)
            new_image = Image.new('RGB', image.size, bg_color)
            new_image.paste(image, (0, 0), image)
            image = new_image

        # Process image and prompt
        inputs = self.processor.process(
            images=[image],
            text=text_prompt,
            return_tensors="pt"
        )

        # Use autocast for memory efficiency
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            pdb.set_trace()
            output = self.model.generate_from_batch(
                batch={k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()},
                generation_config=GenerationConfig(
                    max_new_tokens=200,
                    stop_strings="<|endoftext|>"
                ),
                tokenizer=self.processor.tokenizer
            )

        generated_tokens = output[0, inputs["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Cleanup
        del image, inputs, output, generated_tokens
        torch.cuda.empty_cache()

        return generated_text



    # def generate_response(self, image_path: str, text_prompt: str):
    #     """
    #     Generate a response from the model given an image and a text prompt.
        
    #     :param image_path: Path to the JPG image.
    #     :param text_prompt: Text prompt for the model.
    #     :return: Generated text response.
    #     """
    #     # Load and process image
    #     image = Image.open(image_path).convert("RGB")

    #     # Process inputs
    #     inputs = self.processor.process(
    #         images=[image],
    #         text=text_prompt
    #     )

    #     # Move inputs to model's device
    #     inputs = {k: v.to(self.device).unsqueeze(0) for k, v in inputs.items()}

    #     # Generate response
    #     with torch.no_grad():
    #         output = self.model.generate_from_batch(
    #             inputs,
    #             GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    #             tokenizer=self.processor.tokenizer
    #         )

    #     # Decode generated text
    #     generated_tokens = output[0, inputs["input_ids"].size(1):]
    #     generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    #     return generated_text

    def generate_response(self, image_path: str, text_prompt: str):
        """
        Generate a response from the model given an image and a text prompt.
        Uses bfloat16 and lets Accelerate handle device placement.
        """
        from transformers import GenerationConfig

        # Load and resize image
        image = Image.open(image_path).convert("RGB").resize((224, 224))

        # Process inputs on CPU; do NOT move manually to device
        inputs = self.processor.process(
            images=[image],
            text=text_prompt,
            return_tensors="pt"
        )

        # Prepare batch (batch dim must be outermost)
        batch = {k: v.unsqueeze(0) for k, v in inputs.items()}
        batch["images"] = batch["images"].to(dtype=torch.bfloat16)

        # Generate with autocast + no grad
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                batch=batch,
                generation_config=GenerationConfig(
                    max_new_tokens=200,
                    stop_strings="<|endoftext|>"
                ),
                tokenizer=self.processor.tokenizer
            )

        # Extract generated tokens beyond the prompt
        generated_tokens = output[0, batch["input_ids"].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Free memory
        del image, batch, output, generated_tokens
        torch.cuda.empty_cache()

        return generated_text



    def parse_coordinates(self, output_text: str):
        """
        Parse the output text for coordinates in the format:
        <point x="54.3" y="53.7" alt="teapot">teapot</point>
        
        :param output_text: The text generated by the model.
        :return: Tuple (x, y) as floats if found; otherwise None.
        """
        # Use regex to extract x and y values from the output
        match = re.search(r'<point\s+.*?x="([\d\.]+)"\s+y="([\d\.]+)".*?>', output_text)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            return (x, y)
        else:
            return None