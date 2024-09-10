import json
import unittest

import torch
from transformers import AutoTokenizer

from mllava import conversation as conversation_lib
from mllava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from mllava.train.train import DataArguments, LazySupervisedDataset


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


class TestJais(unittest.TestCase):

    # we ignore the instruction and system text for Jais as they're in English
    # and we're interested in the effects of prompt language on the model

    def test_chat_dataset_processing(self):
        with open('mllava/test/fixtures/en_pretrain_sample.json', 'r') as f:
            data = json.load(f)

        conversation_lib.default_conversation = conversation_lib.conv_templates["jais_chat"]

        tokenizer = AutoTokenizer.from_pretrained(
            'inceptionai/jais-family-6p7b-chat', trust_remote_code=True
        )
        tokenizer.add_tokens(
            [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True
        )

        args = DataArguments(
            is_multimodal=True,
            data_path='mllava/test/fixtures/en_pretrain_sample.json',
            image_folder='mllava/test/fixtures/sampled_images'
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
                '[', '|', 'Human', '|', ']', '<im_patch>', 'Ċ', 'Create', 'Ġa', 'Ġcompact', 'Ġnarrative', 'Ġrepresenting',
                'Ġthe', 'Ġimage', 'Ġpresented', '.',
                'Ġ[', '|', 'AI', '|', ']', 'l', 'arch', 'Ġtrees', 'Ġin', 'Ġautumn', 'Ġcolours', 'Ġalong', 'Ġthe', 'Ġtrail'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )

        self.assertEqual(
            [
                'l', 'arch', 'Ġtrees', 'Ġin', 'Ġautumn', 'Ġcolours', 'Ġalong', 'Ġthe', 'Ġtrail'
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
                '[', '|', 'Human', '|', ']', '<im_patch>', 'Ċ', 'Create', 'Ġa', 'Ġcompact', 'Ġnarrative', 'Ġrepresenting',
                'Ġthe', 'Ġimage', 'Ġpresented', '.',
                'Ġ[', '|', 'AI', '|', ']', 'l', 'arch', 'Ġtrees', 'Ġin', 'Ġautumn', 'Ġcolours', 'Ġalong', 'Ġthe', 'Ġtrail',
                '[', '|', 'Human', '|', ']', 'How', 'Ġtall', 'Ġare', 'Ġthe', 'Ġtrees', '?',
                'Ġ[', '|', 'AI', '|', ']', 'Some', 'Ġare', 'Ġtall', ',', 'Ġsome', 'Ġare', 'Ġshort', '.',
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                'l', 'arch', 'Ġtrees', 'Ġin', 'Ġautumn', 'Ġcolours', 'Ġalong', 'Ġthe', 'Ġtrail',
                'Some', 'Ġare', 'Ġtall', ',', 'Ġsome', 'Ġare', 'Ġshort', '.'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )

        # include a special role token

        processed = dataset[2]
        input_ids = processed['input_ids'].squeeze().tolist()
        input_ids[
            input_ids.index(-200)
        ] = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        labels = [
            label for label in processed['labels'].squeeze().tolist() if label != -100
        ]

        self.assertEqual(
            [
                '[', '|', 'Human', '|', ']', '<im_patch>', 'Ċ', 'Rel', 'ay', 'Ġa', 'Ġbrief', ',', 'Ġclear',
                'Ġaccount', 'Ġof', 'Ġthe', 'Ġpicture', 'Ġshown', '.',
                'Ġ[', '|', 'AI', '|', ']', 'Human', 'Ġin', 'Ġa', 'Ġhall'
            ],
            tokenizer.convert_ids_to_tokens(input_ids)
        )
        self.assertEqual(
            [
                'Human', 'Ġin', 'Ġa', 'Ġhall'
            ],
            tokenizer.convert_ids_to_tokens(labels)
        )
