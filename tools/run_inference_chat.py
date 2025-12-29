import os
import sys
import torch
import spacy
import torch.nn.functional as F
from tqdm import tqdm
from argparse import Namespace, ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local module imports
sys.path.append("../")
try:
    from tools.gpt_llama_templates import GPT_LLAMA_TEMPLATES
    from flags import DATA_FOLDER
    from data.meta_dataset import MetaDataset
except ImportError:
    print("Ensure local modules 'tools', 'flags', and 'data' are in the parent directory.")
    sys.exit(1)

# Dataset prompt templates
CUSTOM_TEMPLATES = {
    "OxfordPets": "the distinctive appearance of a {}, a type of pet.",
    "OxfordFlowers": "the distinctive appearance of a {}, a type of flower.",
    "FGVCAircraft": "the distinctive appearance of a {}, a type of aircraft.",
    "DescribableTextures": "the distinctive appearance of {} texture.",
    "EuroSAT": "the distinctive appearance of a centered satellite photo of {}.",
    "StanfordCars": "the distinctive appearance of a {}.",
    "Food101": "the distinctive appearance of {}, a type of food.",
    "SUN397": "the distinctive appearance of a {}.",
    "Caltech101": "the distinctive appearance of a {}.",
    "UCF101": "the distinctive appearance of a person doing {}.",
    "ImageNet": "the distinctive appearance of a {}.",
    "ImageNetSketch": "the distinctive appearance of a {}.",
    "ImageNetV2": "the distinctive appearance of a {}.",
    "ImageNetA": "the distinctive appearance of a {}.",
    "ImageNetR": "the distinctive appearance of a {}.",
}


def apply_chat_template(tokenizer, sentence):
    """Wraps the input sentence in Llama-3 chat template."""
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": sentence}
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False)


def get_noun_phrases(text, nlp_engine):
    """Extracts noun chunks from generated text using SpaCy."""
    doc = nlp_engine(text)
    return ", ".join([chunk.text for chunk in doc.noun_chunks])


def parse_args():
    parser = ArgumentParser(description="Generate PKV features using Llama-3")
    parser.add_argument('--dataset', type=str, default='FGVCAircraft', help='Dataset name')
    parser.add_argument('--model_path', type=str, default='./meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Path to the local model weights or HuggingFace ID')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def main():
    args_cli = parse_args()

    # Configuration
    args = Namespace(
        dataset=args_cli.dataset,
        seed=args_cli.seed,
        model_path=args_cli.model_path,
    )

    print(f"Current working directory: {os.getcwd()}")

    # Initialize Datasets
    datasets = {
        'base': MetaDataset(dataset=args.dataset, phase='train', seed=args.seed, num_shots=1),
        'new': MetaDataset(dataset=args.dataset, phase='test', seed=args.seed, num_shots=1)
    }

    # Load Model and Tokenizer
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.float16  # Recommended for Llama-3
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    # Setup NLP engine
    try:
        nlp_engine = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spacy model 'en_core_web_sm'...")
        os.system("python -m spacy download en_core_web_sm")
        nlp_engine = spacy.load("en_core_web_sm")

    # Tokenizer configuration
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    dump_dict = {}
    data_dir = datasets['base'].data_dir

    model.eval()

    for target in ['base', 'new']:
        dump_dict[target] = []
        dataset = datasets[target]

        # Iterate through LLM prompt templates defined for this dataset
        for gpt_template in GPT_LLAMA_TEMPLATES[args.dataset]:
            wordfill = [gpt_template.format(c.replace("_", " ")) for c in dataset.classnames]

            # Prepare chat-formatted inputs
            text_input_raw = [apply_chat_template(tokenizer, 'In one sentence, ' + p) for p in wordfill]

            # Append class-specific suffix
            text_inputs_final = [
                raw + f" The {dataset.classnames[idx].replace('_', ' ')} appears as"
                for idx, raw in enumerate(text_input_raw)
            ]

            # Tokenize inputs
            tokenized_inputs = tokenizer(
                text_inputs_final,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            )

            num_pair = tokenized_inputs['input_ids'].shape[0]
            input_ids = tokenized_inputs['input_ids'].cuda()
            attention_mask = tokenized_inputs['attention_mask'].cuda()

            text_predictions = []
            noun_phrase_inputs_list = []

            with torch.inference_mode():
                # Step 1: Generate descriptions
                for i in tqdm(range(0, num_pair, args_cli.batch_size), desc=f"Generating {target}"):
                    batch_ids = input_ids[i: i + args_cli.batch_size]
                    batch_mask = attention_mask[i: i + args_cli.batch_size]

                    output_ids = model.generate(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        do_sample=False,
                        max_new_tokens=128,
                        use_cache=True,
                    )

                    # Decode only the newly generated tokens
                    new_tokens = output_ids[:, batch_ids.shape[1]:]
                    decoded_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                    for text in decoded_texts:
                        text_predictions.append(text)
                        noun_phrase_inputs_list.append(get_noun_phrases(text, nlp_engine))

                # Step 2: Extract PKVs using generated noun phrases
                noun_tokenized = tokenizer(
                    noun_phrase_inputs_list,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                )

                valid_pkv_list = []
                output_attn_mask_list = []

                for i in tqdm(range(0, num_pair, args_cli.batch_size), desc=f"Extracting PKV {target}"):
                    # Concatenate original input tokens with noun phrase tokens
                    batch_input_ids = torch.cat([
                        input_ids[i:i + args_cli.batch_size],
                        noun_tokenized['input_ids'][i:i + args_cli.batch_size].cuda()
                    ], dim=1)

                    batch_attn_mask = torch.cat([
                        attention_mask[i:i + args_cli.batch_size],
                        noun_tokenized['attention_mask'][i:i + args_cli.batch_size].cuda()
                    ], dim=1)

                    extended_inputs = model.prepare_inputs_for_generation(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attn_mask,
                        use_cache=True
                    )

                    outputs = model(**extended_inputs, return_dict=True)

                    # Extract PKV from the last layer [Layers, 2(K/V), Batch, Head, Seq, Dim]
                    # We store only the last layer's KV cache as per original implementation
                    pkv = torch.stack([
                        torch.stack([kv.cpu() for kv in layer_kvs])
                        for layer_kvs in outputs.past_key_values[-1:]
                    ])

                    valid_pkv_list.append(pkv)
                    output_attn_mask_list.append(batch_attn_mask.cpu())

            # Combine batches
            combined_pkv = torch.cat(valid_pkv_list, dim=2)
            combined_mask = torch.cat(output_attn_mask_list, dim=0)

            dump_dict[target].append({
                'past_key_values': combined_pkv,
                'attn_mask': combined_mask,
                'text_predictions': text_predictions
            })

    # Save results
    output_file = os.path.join(data_dir, 'llama3_dw_past_key_value.pt')
    torch.save(dump_dict, output_file)
    print(f"Features saved successfully to {output_file}")


if __name__ == "__main__":
    main()