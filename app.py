import gradio as gr
import torch
import os

from PIL import Image
from pathlib import Path
from more_itertools import chunked

from transformers import CLIPProcessor, CLIPModel

checkpoint = "openai/clip-vit-base-patch32"
x_, _, files = next(os.walk("./emojis"))
no_of_emojis = range(len(files))
emojis_as_images = [Image.open(f"emojis/{i}.png") for i in no_of_emojis]
K = 4

processor = CLIPProcessor.from_pretrained(checkpoint)
model = CLIPModel.from_pretrained(checkpoint)


def concat_images(*images):
    """Generate composite of all supplied images."""
    # Get the widest width.
    width = max(image.width for image in images)
    # Add up all the heights.
    height = max(image.height for image in images)
    # set the correct size of width and heigtht of composite.
    composite = Image.new('RGB', (2*width, 2*height))
    assert K == 4, "We expect 4 suggestions, other numbers won't work."
    for i, image in enumerate(images):
        if i == 0:
            composite.paste(image, (0, 0))
        elif i == 1:
            composite.paste(image, (width, 0))
        elif i == 2:
            composite.paste(image, (0, height))
        elif i == 3:
            composite.paste(image, (width, height))
    return composite


def get_emoji(text, model=model, processor=processor, emojis=emojis_as_images, K=4):
    inputs = processor(text=text, images=emojis, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)

    logits_per_text = outputs.logits_per_text
    # we take the softmax to get the label probabilities
    probs = logits_per_text.softmax(dim=1)
    # top K number of options
    predictions_suggestions_for_chunk = [torch.topk(prob, K).indices.tolist() for prob in probs][0]
    # We can get this as a response from the backend
    predictions_suggestions_for_chunk

    images = [Image.open(f"emojis/{i}.png") for i in predictions_suggestions_for_chunk]
    images_concat = concat_images(*images)
    return images_concat


text = gr.inputs.Textbox(placeholder="Enter a text and I predict an emoji...")
title = "Predicting an Emoji"
description = """Please provide a sentence and I will suggest 4 from the following emoji's:
\n❤️ 😍 😂 💕 🔥 😊 😎 ✨ 💙 😘 📷 🇺🇸 ☀ 💜 😉 💯 😁 🎄 📸 😜 ☹️ 😭 😔 😡 💢 😤 😳 🙃 😩 😠 🙈 🙄\n
You can also try out the examples at the bottom.
"""

examples = [
    "I'm so happy for you!",
    "I'm not feeling great today.",
    "This makes me angry!",
    "Can I follow you?",
    "I'm so bored right now ...",
]
gr.Interface(fn=get_emoji, inputs=text, outputs=gr.Image(shape=(24,24)), 
             examples=examples, title=title, description=description
             ).launch()