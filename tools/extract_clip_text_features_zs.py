import os
import re
import sys
import torch
import spacy
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import CLIPModel, CLIPProcessor
from collections import defaultdict

# Add parent directory to path for local imports
sys.path.append("../")

from data.meta_dataset import MetaDataset

# Configuration for different datasets
CUSTOM_TEMPLATES = {
    "OxfordPets": "a type of pet, a photo of a {}.",
    "OxfordFlowers": "a type of flower, a photo of a {}.",
    "FGVCAircraft": "a type of aircraft, a photo of a {}.",
    "DescribableTextures": "a texture of {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a type of food, a photo of {}.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def clean_text(text):
    """Clean text by removing extra spaces and quotes."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '').replace("'", "")
    return text


def parse_args():
    parser = ArgumentParser(description="Generate CLIP embeddings and prompts.")
    parser.add_argument('--dataset', type=str, default='FGVCAircraft', help='Dataset name')
    parser.add_argument('--clip_path', type=str, default='openai/clip-vit-base-patch16',
                        help='Path to local CLIP model or HuggingFace ID')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set environment variables before torch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Current working directory: {os.getcwd()}")

    # Initialize Dataset
    datasets = {
        'base': MetaDataset(dataset=args.dataset, phase='train', seed=args.seed, num_shots=1),
        'new': MetaDataset(dataset=args.dataset, phase='test', seed=args.seed, num_shots=1)
    }

    # Load Model and Processor
    print(f"Loading CLIP from: {args.clip_path}")
    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    nlp_engine = spacy.load("en_core_web_sm")

    output_embeddings = {}
    templates_dict = {}
    data_dir = datasets['base'].data_dir
    cur_template = CUSTOM_TEMPLATES.get(args.dataset, "a photo of a {}.")

    suffixes = ['aw']
    base_filename = 'llama3_{}_past_key_value.pt'
    save_filename = 'llama3_total_clip_text_embeddings.pt'

    for suffix in tqdm(suffixes, desc="Processing suffixes"):
        pkv_file_path = base_filename.format(suffix)
        templates_dict[suffix] = {}
        output_embeddings[suffix] = {}

        for split in ['base', 'new']:
            dataset = datasets[split]
            all_text_embeddings = []
            temp_class_embeddings = []
            templates_dict[suffix][split] = {}

            for i, class_name in tqdm(enumerate(dataset.classnames),
                                      total=len(dataset.classnames),
                                      desc=f"Processing {split} classes"):

                noun_chunks = []
                # Check if we use the default 'all-weather' template or LLM-generated features
                if '_aw_' in pkv_file_path:
                    base_text = cur_template.format(class_name.replace("_", " "))
                else:
                    content_path = os.path.join(data_dir, pkv_file_path)
                    content_dict = torch.load(content_path, map_location='cpu')
                    # Use the first LLM prediction available
                    prediction = content_dict[split][0]['text_predictions'][i]
                    doc = nlp_engine(prediction)
                    noun_chunks = list(doc.noun_chunks)
                    base_text = cur_template.format(class_name.replace("_", " "))

                all_feat_embeds = []
                display_text = ""

                with torch.inference_mode():
                    if '_aw_' in pkv_file_path:
                        display_text = base_text
                        full_input = processor(text=base_text, return_tensors="pt",
                                               padding=True, max_length=128, truncation=True).to(device)
                        text_embeds = F.normalize(clip_model.get_text_features(**full_input), dim=-1)
                        temp_class_embeddings.append(text_embeds[0])
                    else:
                        if not noun_chunks:
                            display_text = f"{base_text.rstrip('.')} from the {args.dataset} dataset."
                            full_input = processor(text=base_text, return_tensors="pt",
                                                   padding=True, max_length=128, truncation=True).to(device)
                            text_embeds = F.normalize(clip_model.get_text_features(**full_input), dim=-1)
                            all_feat_embeds.append(text_embeds)
                        else:
                            display_text = f"{base_text.rstrip('.')} with {prediction}"
                            for n in noun_chunks:
                                full_text = f"{base_text.rstrip('.')} with {n.text}"
                                full_input = processor(text=full_text, return_tensors="pt",
                                                       padding=True, max_length=128, truncation=True).to(device)
                                text_embeds = F.normalize(clip_model.get_text_features(**full_input), dim=-1)
                                all_feat_embeds.append(text_embeds)

                        # Average embeddings for all noun chunks of a class
                        avg_embed = torch.cat(all_feat_embeds, dim=0).mean(dim=0)
                        temp_class_embeddings.append(avg_embed)

                unique_key = f"{class_name}_{i}" if args.dataset == 'ImageNet' else class_name
                templates_dict[suffix][split][unique_key] = clean_text(display_text)

            # Stack all class embeddings for this split
            stacked_embeddings = torch.stack(temp_class_embeddings, dim=0)
            output_embeddings[suffix][split] = {
                'avg': stacked_embeddings,
                'all': stacked_embeddings.unsqueeze(0)
            }

    # Save embeddings
    save_path = os.path.join(data_dir, save_filename)
    torch.save(output_embeddings, save_path)
    print(f"Embeddings saved to {save_path}")

    # Generate template Python file
    save_templates_to_file(args.dataset, suffixes, templates_dict)


def save_templates_to_file(dataset_name, suffixes, templates_dict):
    """Saves the generated text prompts into a organized python file."""
    output_content = f"{dataset_name}_TEMPLATES = {{\n"
    for suffix in suffixes:
        output_content += f"    {repr(suffix)}: {{\n"
        for split in ['base', 'new']:
            output_content += f"        {repr(split)}: {{\n"
            for key, text in templates_dict[suffix][split].items():
                output_content += f"            {repr(key)}: {repr(text)},\n"
            output_content += "        },\n"
        output_content += "    },\n"
    output_content += "}\n"

    # Save to kolors_prompt directory
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'kolors_prompt')
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, f"{dataset_name.lower()}_kpro.py")

    with open(file_path, 'w') as f:
        f.write(output_content)
    print(f"Templates saved to {file_path}")


if __name__ == "__main__":
    main()