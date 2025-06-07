import os
import json
import time
import base64
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger
from openai import OpenAI
from argparse import ArgumentParser

load_dotenv()
parser = ArgumentParser()
parser.add_argument('--router', type=str, default='OpenRouter')
parser.add_argument('--model', type=str, default='google/gemma-3-4b-it:free')
parser.add_argument('--input_file', type=str, default='mini.json')
parser.add_argument('--output_file', type=str, default='prediction.json')
args = parser.parse_args()

client=OpenAI(api_key='<KEY>')
if args.router == 'OpenRouter':
    client.api_key=os.getenv('OPENROUTER_API_KEY')
    client.base_url='https://openrouter.ai/api/v1'
elif args.router == 'AIStudio':
    client.api_key=os.getenv('AISTUDIO_API_KEY')
    client.base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
elif args.router == 'OpenAI':
    client.api_key=os.getenv('OPENAI_API_KEY')
    client.base_url='https://api.openai.com/v1/'
elif args.router == 'Anthropic':
    client.api_key=os.getenv('ANTHROPIC_API_KEY')
    client.base_url='https://api.anthropic.com/v1/'

def build_prompt(item):
    tgt_path = item['image_path']
    question = item['question']
    if item['language'] == 'English':
        question += "\nPlease answer this question with reasoning. First output your reasoning process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
    else:
        question += "\n请用推理来回答这个问题。首先在<think></think>标签中输出推理过程，然后在<answer></answer>标签中输入最终答案。"

    try:
        if item['sig_figs']:
            sf = str(int(item['sig_figs']))
            if item['language'] == 'English':
                question += f"The final answer should retain {sf} significant figures."
            else:
                question += f"最终答案应保留{sf}位有效数字。"
    except Exception as e:
        pass

    return question, tgt_path   


def encode_image(image_path):
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


def run_inference(json_path, output_path, model):
    results = []  
    with open(json_path, 'r', encoding='utf-8') as f:  
        data = json.load(f)
    
    for idx, input_item in enumerate(tqdm(data)):
        question, img_paths= build_prompt(input_item)
        base64_images = [encode_image(img_path) for img_path in img_paths]
        response = ""  
        print(question)
        max_retries = 5
        retry_delay = 30
        attempt = 0

        while attempt < max_retries:
            try:
                response = inference_one_step(question, base64_images, model)
                break
            except Exception as e:
                attempt += 1
                logger.error(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    logger.info(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Skipping this item.")
                    response = "ERROR: Max retries reached."
            
        results.append({  
            "index": input_item['index'],
            "question": question,
            "subject": input_item['subject'],
            "img_category": input_item['img_category'],
            "vision_relevance": input_item['vision_relevance'],
            "language": input_item['language'],
            "level": input_item['level'],
            "sig_figs": input_item['sig_figs'],
            "caption": input_item['caption'],
            "prediction": response,
        })
     
        with open(output_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

def inference_one_step(question, base64_images, model):
    payload = client.chat.completions.create(
                model=model,
                messages=[
                    {   
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            }
                        ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        }
                } for base64_image in base64_images
            ]
                    },
                ],
                max_tokens=64000,
            )
    response = payload.choices[0].message.content
    print(response)
    return response

if __name__ == '__main__':
    # 'dev.json', 'total.json'
    run_inference(args.input_file, args.output_file, model=args.model)