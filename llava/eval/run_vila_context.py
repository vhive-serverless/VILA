import argparse
import os
import os.path as osp
import re
from io import BytesIO
import csv

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
    opencv_extract_frames,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def save_sampled_frames(images, output_dir="output_frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, image in enumerate(images):
        image_path = os.path.join(output_dir, f"frame_{idx+1}.jpg")
        image.save(image_path)
        print(f"Saved frame {idx+1} to {image_path}")

def eval_model(args, model, tokenizer, image_processor, context_len, images):
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("No <image> tag found in input. Automatically appending one at the beginning of the text.")
            # Do not repeatedly append the prompt.
            if model.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
    print("Input:", qs)

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if args.temperature > 0 else False,
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
    print("Output:", outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-file", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=20)
    parser.add_argument("--conv-mode", type=str, default=None)
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

    # Load video and extract frames
    if args.video_file.startswith("http") or args.video_file.startswith("https"):
        print("Downloading video from URL:", args.video_file)
        response = requests.get(args.video_file)
        video_file = BytesIO(response.content)
    else:
        assert osp.exists(args.video_file), "Video file not found"
        video_file = args.video_file

    images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)

    # Split images into two halves
    # half_point = len(images) // 2
    images_first_half = images[:5]
    images_second_half = images[13:]

    save_sampled_frames(images_first_half, output_dir="extracted_frames_first_half")
    save_sampled_frames(images_second_half, output_dir="extracted_frames_second_half")

    # Process first half
    args.query = '<video>\nThe video is extracted from a street camera, describe briefly what is happening in the video.'
    output_first_half = eval_model(args, model, tokenizer, image_processor, context_len, images_first_half)

    # Process second half with context from the first half
    args.query = f'<video>\nBased on the description of normal condition: "{output_first_half}", describe whether there is an accident in the following video.'
    output_second_half = eval_model(args, model, tokenizer, image_processor, context_len, images_second_half)

    # Write outputs to CSV
    with open('output_context.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.stat('output_context.csv').st_size == 0:
            writer.writerow(["video_file", "first_half_output", "second_half_output"])
        video_name = os.path.splitext(os.path.basename(args.video_file))[0]
        writer.writerow([video_name, output_first_half, output_second_half])
