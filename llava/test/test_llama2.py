import unittest

import torch
from transformers import AutoTokenizer

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from llava.train.train import DataArguments, LazySupervisedDataset


class DummyProcessor:
    def __init__(self):
        self.crop_size = 224

    def preprocess(self, _, return_tensors):
        assert return_tensors
        return {
            'pixel_values': torch.randn(
                1, 3, 224, 224
            )
        }


class TestBaichuan(unittest.TestCase):

    # https://github.com/meta-llama/llama/blob/main/llama/generation.py#L343
    #   f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
    #       - adds bos, eos after all prior instructions and final output

    def test_chat_dataset_processing(self):
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama_2_chat"]

        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens(
            [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )

        args = DataArguments(
            is_multimodal=True,
            data_path='llava/test/fixtures/en_pretrain_sample.json',
            image_folder='llava/test/fixtures/sampled_images'
        )
        args.mm_use_im_start_end = False
        args.image_processor = DummyProcessor()

        dataset = LazySupervisedDataset(
            args.data_path, tokenizer, args
        )

        self.assertEqual(len(dataset), 5)

        # two turns

        processed = dataset[0]
        input_ids = processed['input_ids'].squeeze().tolist()
        input_ids[
            input_ids.index(-200)
        ] = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        labels = [
            label for label in processed['labels'].squeeze().tolist() if label != -100
        ]

        self.assertEqual(
            [
                '<s>', '▁[', 'INST', ']', '▁', '<im_patch>', '▁', '<0x0A>', 'Create', '▁a', '▁compact', '▁narr',
                'ative', '▁representing', '▁the', '▁image', '▁presented', '.', '▁[', '/', 'INST', ']',
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )

        # four turns

        processed = dataset[1]
        input_ids = processed['input_ids'].squeeze().tolist()
        input_ids[
            input_ids.index(-200)
        ] = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        labels = [
            label for label in processed['labels'].squeeze().tolist() if label != -100
        ]

        self.assertEqual(
            [
                '<s>', '▁[', 'INST', ']', '▁', '<im_patch>', '▁', '<0x0A>', 'Create', '▁a', '▁compact', '▁narr', 'ative',
                '▁representing', '▁the', '▁image', '▁presented', '.', '▁[', '/', 'INST', ']',
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>',
                '<s>', '▁[', 'INST', ']', '▁How', '▁tall', '▁are', '▁the', '▁trees', '?', '▁[', '/', 'INST', ']',
                '▁Some', '▁are', '▁tall', ',', '▁some', '▁are', '▁short', '.', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                '▁l', 'arch', '▁trees', '▁in', '▁aut', 'umn', '▁colours', '▁along', '▁the', '▁trail', '▁', '</s>',
                '▁Some', '▁are', '▁tall', ',', '▁some', '▁are', '▁short', '.', '▁', '</s>'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )
