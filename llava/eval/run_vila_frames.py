import argparse
import os
import os.path as osp
import re
from io import BytesIO
import csv
import json
import time  # Import time to measure inference duration

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


def generate_output(qs, images_tensor, model, tokenizer, args):
    # Prepare conversation prompt
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Ensure the number of image tokens matches the number of images
    num_image_tokens = prompt.count(DEFAULT_IMAGE_TOKEN)
    if num_image_tokens != images_tensor.size(0):
        raise ValueError(
            f"The number of image tokens ({num_image_tokens}) does not match the number of images ({images_tensor.size(0)})."
        )

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    # Define stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(f"Running model for situation:\n{qs}")
    with torch.inference_mode():
        # Measure the start time
        start_time = time.time()

        # Generate output from model
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

        # Measure the end time
        end_time = time.time()

    # Calculate the inference time
    inference_time = end_time - start_time

    # Decode and return the model's output and inference time
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(f"Model Output:\n{outputs}")
    print(f"Inference Time: {inference_time:.2f} seconds")
    return outputs, inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=50)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--situations", type=str, nargs='*', default=None,
                        help="Specify which situations to run. Provide as space-separated list.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing entries in the JSON file.")
    args = parser.parse_args()

    # Define the number of frames to analyze
    for length in range(5, 21):

        # Initialize model once
        disable_torch_init()
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path, model_name, args.model_base
        )

        # Define the list of all possible situations
        all_situations = [
            "Are there any collisions between vehicles, pedestrians or other objects? ",
            "Are there any overturned or heavily damaged vehicles? ",
            "Are there any signs of sudden braking or skidding by any vehicle? ",
            "Are there any obvious smoke, fire, or debris in the scene? ",
            "Are there any people appear to be hit or falling? ",
        ]

        # Determine which situations to run
        if args.situations:
            # Map input strings to the corresponding situations
            situation_map = {str(i+1): s for i, s in enumerate(all_situations)}
            selected_situations = []
            for s in args.situations:
                if s in situation_map:
                    selected_situations.append(situation_map[s])
                elif s in all_situations:
                    selected_situations.append(s)
                else:
                    print(f"Warning: Situation '{s}' not recognized.")
            if not selected_situations:
                raise ValueError("No valid situations specified.")
        else:
            selected_situations = all_situations

        # Query to append if the answer is true
        query = (
            "Please provide a \"Yes\" or \"No\" answer.\nIf the answers is \"Yes\": \n" 
            "The situation is seen on which given frame? \n"
            "Example: Yes, the situation happened from / No.\n"
            # "- The position of this situation in the frame? \"A. left\", \"B. right\", \"C. center\"."
        )

        # Load video frames
        if args.video_file is None:
            raise ValueError("No video file provided")
        else:
            if args.video_file.startswith("http"):
                print("Downloading video from URL:", args.video_file)
                response = requests.get(args.video_file)
                video_file = BytesIO(response.content)
            else:
                assert osp.exists(args.video_file), "Video file not found"
                video_file = args.video_file

            from llava.mm_utils import opencv_extract_frames

            # Extract frames from the video
            images, num_frames = opencv_extract_frames(video_file, args.num_video_frames)


        # Create a folder to store extracted frames
        video_name = osp.splitext(osp.basename(args.video_file))[0]
        print(video_name)
        folder_name = f"{num_frames}/{video_name}"
        output_folder = osp.join("./frames", folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # Save extracted frames to the folder
        for i, image in enumerate(images):
            image_path = osp.join(output_folder, f"frame_{i + 1}.jpg")
            image.save(image_path)
            print(f"Saved frame {i + 1} to {image_path}")

        # Select a subset of frames for analysis
        start = 30
        images = images[start:(start+length)]

        # Build the prompt with all frames
        qs_frames = ""
        for i in range(len(images)):
            qs_frames += f"Frame {i + start + 1}: {DEFAULT_IMAGE_TOKEN}\n"

        # Process images
        images_tensor = process_images(images, image_processor, model.config).to(
            model.device, dtype=torch.float16
        )

        # Load existing results if the JSON file exists
        results_file = f'results_range_{length}.json'
        if osp.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            results = []

        # Convert results to a dict for easy lookup
        results_dict = {(r['video'], r['situation']): r for r in results}

        # List to store inference times
        inference_times = []

        for situation in selected_situations:
            # Build the prompt for each situation
            qs = qs_frames + "Analyze the given frames and answer the following questions:\n" + situation + "\n" + query

            # Check if the result already exists
            key = (video_name, situation)
            if key in results_dict and not args.overwrite:
                print(f"Result for video '{video_name}', situation '{situation}' already exists. Skipping.")
                continue

            # Generate output for the current situation
            output, inference_time = generate_output(qs, images_tensor, model, tokenizer, args)

            # Collect the output and inference time
            result = {
                "video": video_name,
                "situation": situation,
                "output": output,
            }

            # Update or append the result
            results_dict[key] = result

            # Add the inference time to the list
            inference_times.append(inference_time)

        # Calculate and display the average inference time
        if inference_times:
            average_time = sum(inference_times) / len(inference_times)
            print(f"\nAverage Inference Time per Situation: {average_time:.2f} seconds")
        else:
            print("\nNo new inferences were performed.")

        # Convert results_dict back to a list
        results = list(results_dict.values())

        # Store the results as a JSON file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
