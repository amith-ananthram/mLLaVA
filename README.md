# mLLaVA  

A fork of the [LLaVA codebase](https://github.com/haotian-liu/LLaVA/tree/main) with added support for using the Baichuan2 models as the base LLM.  This codebase is a dependency of the experiments in [See It From My Perspective](https://github.com/amith-ananthram/see-it-from-my-perspective/tree/main).

## Models

Models were trained using the v1.5 [pretraining](https://github.com/amith-ananthram/mLLaVA/blob/main/scripts/v1_5/pretrain.sh) and [LORA fine-tuning](https://github.com/amith-ananthram/mLLaVA/blob/main/scripts/v1_5/finetune_lora.sh) scripts.  We use the original LLaVA v1 fusion corpus (for Chinese, we use [the translation shared by LinkSoul](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)). 

Each model directory is around 600MB unzipped as they only contain 1) the projector and 2) the LORA weights.  The codebase will download the rest of the model (Llama2-7B or Baichuan2-7B-Chat and CLIP-L).

| Base LLM | Fusion Corpus Language(s) | Download Link |
| ------- | ------- | ------ |
| Llama2-7B-Chat | en | link |
| | zh | link |
| | en/zh | link |
| Baichuan2-7B-Chat | en | link |
| | zh | link |
| | en/zh | link |

If you use any of these models, please cite:

1) [Visual Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html), Liu et al., 2024
2) [See It from My Perspective: Diagnosing the Western Cultural Bias of Large Vision-Language Models in Image Understanding](https://arxiv.org/abs/2406.11665), Ananthram et al., 2024 

And https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions.

## Example Usage

    import torch
    from PIL import Image

    from mllava.mm_utils import get_model_name_from_path 
    from mllava.model.builder import load_pretrained_model 
    from mllava.conversation import conv_templates
    from mllava.constants import IMAGE_TOKEN_INDEX
    from mllava.mm_utils import tokenizer_image_token 
    from mllava.mm_utils import process_images 

    device = torch.device('cuda')

    model_path = 'path/to/model/dir'
    if 'baichuan' in model_path:
      conv_mode = 'baichuan_2_chat'
      model_base = 'baichuan-inc/Baichuan2-7B-Chat'
    else:
      assert 'llama' in model_path
      conv_mode = 'llama_2_chat'
      model_base = 'meta-llama/Llama-2-7b-chat-hf'

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
      model_path, model_base, model_name, device=device
    )

    image_path = 'path/to/image/file'
    query = '<image>\nPlease describe this image.'

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], template)
    conv.append_message(conv.roles[1], None)
    template = conv.get_prompt()
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).squeeze()
    processed = {
        'input_ids': input_ids,
        'attention_mask': torch.ones(input_ids.shape[0])
    }

    with Image.open(image_path) as img:
        images = process_images(
            [img.convert('RGB')], image_processor, model.config
        ).to(dtype=torch.float16).squeeze()
        image_sizes = img.size

    output = model.generate(
      inputs=processed['input_ids'].unsqueeze(dim=0).to(device),
      attention_mask=processed['attention_mask'].unsqueeze(dim=0).to(device),
      images=images.unsqueeze(dim=0).to(device),
      image_sizes=[image_sizes]
    )
    
