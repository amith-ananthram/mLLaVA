import sys
import torch
from PIL import Image

from loader import load_pretrained_model
from conversation import conv_templates
from constants import IMAGE_TOKEN_INDEX
from utils import get_model_name_from_path, tokenizer_image_token, process_images


if __name__ == '__main__':
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    prompt = sys.argv[3]
    hub_name = sys.argv[4]

    device = torch.device('cuda')

    if 'baichuan' in model_path:
        conv_mode = 'baichuan_2_chat'
        model_base = 'baichuan-inc/Baichuan2-7B-Chat'
    else:
        assert 'llama' in model_path
        conv_mode = 'llama_2_chat'
        model_base = 'meta-llama/Llama-2-7b-chat-hf'

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, is_lora=True
    )
    print("Model name: %s" % model_name)

    if hub_name.strip() != "":
        model.push_to_hub(hub_name.strip())

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], '<image>\n' + prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).squeeze()

    with Image.open(image_path) as img:
        images = process_images(
            [img.convert('RGB')], image_processor, model.config
        ).to(dtype=torch.float16).squeeze()
        image_sizes = img.size

    output = model.generate(
        inputs=input_ids.unsqueeze(dim=0).to(device),
        attention_mask=torch.ones(input_ids.shape[0]).unsqueeze(dim=0).to(device),
        images=images.unsqueeze(dim=0).to(device),
        image_sizes=[image_sizes]
    )

    print(tokenizer.batch_decode(output, skip_special_tokens=True))
