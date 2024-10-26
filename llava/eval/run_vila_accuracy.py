# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import os
import os.path as osp
import re
from io import BytesIO
import csv
from PIL import Image

import requests
import torch

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args, model, tokenizer, image_processor, context_len):
    # Load video frames
    if args.video_file is None:
        image_files = image_parser(args)
        images = load_images(image_files)
    else:
        if args.video_file.startswith("http"):
            print("Downloading video from URL:", args.video_file)
            response = requests.get(args.video_file)
            video_file = BytesIO(response.content)
        else:
            assert osp.exists(args.video_file), "Video file not found"
            video_file = args.video_file

        from llava.mm_utils import opencv_extract_frames

        images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("No <image> tag found in input. Automatically appending one at the beginning of text.")
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

    print("Input:", qs)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[images_tensor],
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=10)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--query", type=str, default="What does the image show?"
                        # "Are there any collisions between vehicles, pedestrians, or obstacles?\n"
                        # "Is there any overturned or heavily damaged vehicles?  Is there any signs of sudden braking or skidding by any vehicle?\n"
                        # "Identify any smoke, fire, or debris on the road.\n"
                        # "Identify if any pedestrians or cyclists appear to be in danger or falling.\n"
                        # "Identify any unusual vehicle positions or movements.\n"
                        # "If the answer is true, provide the following information:"
                        # "- The frame number in which the collision first occurs."
                        # "- The approximate location of the collision in that frame (e.g., left, right, center, top, bottom)."
                        )
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Initialize model once
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, model_name, args.model_base
    )

    # with open("output.csv", mode="a", newline="") as file:
    #     writer = csv.writer(file)

    #     if os.stat("output.csv").st_size == 0:
    #         writer.writerow(["video_file", "num_video_frames", "outputs"])

    # video_name = os.path.splitext(os.path.basename(args.video_file))[0]

    outputs = eval_model(args, model, tokenizer, image_processor, context_len)

    # writer.writerow([video_name, args.num_video_frames, outputs])