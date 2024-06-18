# mLLaVA  

A fork of the [LLaVA codebase](https://github.com/haotian-liu/LLaVA/tree/main) with added support for using the Baichuan2 models as the base LLM.  This codebase is a dependency of the experiments in [See It From My Perspective](https://github.com/amith-ananthram/see-it-from-my-perspective/tree/main).

## Models

Models were trained using the v1.5 [pretraining](https://github.com/amith-ananthram/mLLaVA/blob/main/scripts/v1_5/pretrain.sh) and [LORA fine-tuning](https://github.com/amith-ananthram/mLLaVA/blob/main/scripts/v1_5/finetune_lora.sh) scripts.  We use the original LLaVA v1 fusion corpus (for Chinese, we use [the translation shared by LinkSoul](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions)). 

Each model directory is around 600MB unzipped as they only contain 1) the projector and 2) the LORA weights.  The codebase will download the rest of the model (Llama2-7B or Baichuan2-7B-Chat and CLIP-L).

| Base LLM | Fusion Corpus Language(s) | Download Link |
| ------- | ------- | ------ |
| Llama2-7B-Chat | en | [link](https://drive.google.com/file/d/16H18ZwmHUNMmCNXinP0_kRdwNBYmJjg1/view?usp=sharing) |
| | zh | [link](https://drive.google.com/file/d/1ScU9Xhstn5mtXN0et4X8LATxoOBxkTZi/view?usp=sharing) |
| | en/zh | [link](https://drive.google.com/file/d/15fYREDYbzT8VJlRaXeb9A-bc6TRnGCbo/view?usp=sharing) |
| Baichuan2-7B-Chat | en | [link](https://drive.google.com/file/d/1IH6TfObGf3wAXsgmSWtMvka80gc5J35C/view?usp=sharing) |
| | zh | [link](https://drive.google.com/file/d/1iI2G7p0zTQUvaoVC9wqet-vwwq0sVD0Y/view?usp=sharing) |
| | en/zh | [link](https://drive.google.com/file/d/1ym3JB66gsCJrH7fobG2P3kw-_f9RADY9/view?usp=sharing) |

If you use any of these models, please cite:

1) [Visual Instruction Tuning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html), Liu et al., 2024
2) [See It from My Perspective: Diagnosing the Western Cultural Bias of Large Vision-Language Models in Image Understanding](https://arxiv.org/abs/2406.11665), Ananthram et al., 2024 

And https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions.

## Example Usage

For a minimal example of using one of our models for inference, please see [predict.py](https://github.com/amith-ananthram/mLLaVA/blob/main/predict.py).
    
