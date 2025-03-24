#!/usr/bin/env python
# coding: utf-8

# # Creating Dog Picture Stories with LLaVA and Llama 3.1

# [LLaVA](https://llava-vl.github.io/) (**L**arge **L**anguage-**a**nd-**V**ision **A**ssistant) is a powerful vision model that combines the capabilities of Large Language Models (LLMs) with image analysis. This open-source model can answer visual questions, generate captions, and perform Optical Character
# Recognition (OCR), making it an ideal solution for applications that require image-based text generation. The `llava-v1.5-7b-4096`model is now available as a preview on [Groq API](https://console.groq.com/docs/vision), allowing for integration of advanced vision capabilities into applications to unlock new possibilities for image-based text generation.

# In this tutorial, we'll explore the capabilities of LLaVA by analyzing a dataset of dog pictures. We'll demonstrate how to use the model to generate text descriptions of the images, and then use the `llama-3.1-70b-versatile` model powered by Groq to write short children's stories based on the images provided. By the end of this tutorial, you'll be able to use LLaVA to generate text descriptions of images and create engaging stories based on those descriptions.

# ### Setup

# Import packages
import base64

import matplotlib.pyplot as plt
from PIL import Image

from groq import Groq


# This tutorial requires a Groq API key - if you don't already have one, you can create one a free GroqCloud account [here](https://console.groq.com) to generate a Groq API Key. We will be using the `llava-v1.5-7b-4096-preview` model for image descriptions and the `llama-3.1-70b-versatile` model for storytelling.

client = Groq()
# llava_model = "llava-v1.5-7b-4096-preview"  # deprecated
llava_model = "llama-3.2-11b-vision-preview"
# llama31_model = "llama-3.1-70b-versatile"  # deprecated
llama31_model = "llama-3.3-70b-versatile"


# ### Basic LLaVA usage

# For this tutorial, we'll use a set of dog images from [Britannica](https://www.britannica.com/animal/dog). You can find the images in the [Groq API cookbook repository](https://github.com/groq/groq-api-cookbook/tree/main/tutorials/llava-image-processing/images).
#
# We'll be using a local image and encoding it to base64 to use with LLaVA. This is one of two ways to use an image with LLaVA - the other is to provide the actual URL of the image.

# Load and display image
image_path = "images/labradoodle.png"
img = Image.open(image_path)

plt.imshow(img)
plt.axis("off")
plt.show()


# To use locally saved images with LLaVA, we'll need to encode them to base64 first:


# Define image encoding function
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


base64_image = encode_image(image_path)


# To use LLaVA with the Groq API, we'll make a request to the `https://api.groq.com/openai/v1/chat/completions` API endpoint. However, when working with images, we need to pass the image and prompt in a slightly different format than what you might be used to. Instead of just passing a text prompt, we need to wrap it in a special JSON structure that includes the image as well. This allows LLaVA to understand that the image is part of the prompt and generate a response accordingly:


# Define image to text function
def image_to_text(client, model, base64_image, prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


# ### Image Storytelling with Llama 3.1

# Now, let's take our image recognition to the next level by creating short stories based on the images used. Since LLaVA is great at broadly describing what's in an image, we will use LLaVA to describe the image and Llama 3.1 70B to write a short children's story based on the image description:


# Define short story generation function
def short_story_generation(client, image_description):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a children's book author. Write a short story about the scene depicted in this image or images.",
            },
            {
                "role": "user",
                "content": image_description,
            },
        ],
        model=llama31_model,
    )

    return chat_completion.choices[0].message.content


prompt = """
Describe this image in detail, including the appearance of the dog(s) and any notable actions or behaviors.
"""
image_description = image_to_text(client, llava_model, base64_image, prompt)

print(short_story_generation(client, image_description))


# Now, let's generate descriptions for multiple images and create a story with both of them. Note that this will have to be done using multiple LLaVA calls as the preview does not currently support multi-image uploads:

base64_image1 = encode_image("images/husky.png")
base64_image2 = encode_image("images/bulldog.png")

image_description1 = image_to_text(client, llava_model, base64_image1, prompt)
image_description2 = image_to_text(client, llava_model, base64_image2, prompt)

combined_image_description = image_description1 + "\n\n" + image_description2

print(short_story_generation(client, combined_image_description))


# ### Conclusion

# In this tutorial, we've explored the capabilities of LLaVA and Llama 3.1 powered by Groq for lightning-fast inference speed in generating text descriptions of images and creating short stories based on those descriptions. We've seen how to use LLaVA to describe images in detail, and how to use Llama 3.1 to write engaging stories based on those descriptions. By combining these two models hosted on GroqCloud, we can create a powerful tool for generating text-based content from images. Give it a try today!
