# -*- coding: utf-8 -*-
"""
Validation script with MeaCap Retrieve-then-Filter module
This version replaces ViECap's original entity detection with MeaCap's Retrieve-then-Filter module
"""

import os
import json
import pickle
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import clip
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
from utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP
from search import beam_search, greedy_search, opt_search

import copy
cpu_device = torch.device('cpu')


def validation_coco_flickr30k_meacap(
    args,
    inpath: str,                             # path of annotations file
    model: ClipCaptionModel,                 # trained language model
    tokenizer: AutoTokenizer,                # tokenizer 
    vl_model: CLIP,                          # CLIP model for retrieval
    vl_model_retrieve: CLIP,                 # CLIP model for retrieval (may be on CPU)
    memory_captions: list,                   # memory bank captions
    memory_clip_embeddings: torch.Tensor,     # memory bank CLIP embeddings
    memory_wte_embeddings: torch.Tensor,     # memory bank SentenceBERT embeddings
    parser_model,                            # Flan-T5 parser model
    parser_tokenizer,                        # Flan-T5 parser tokenizer
    wte_model: SentenceTransformer,          # SentenceBERT model
    retrieve_on_CPU: bool,                   # whether to retrieve on CPU
    preprocess: clip = None,                 # processor of the image
    encoder: clip = None,                    # clip backbone
) -> None:
    """
    Validation function for COCO/Flickr30k using MeaCap Retrieve-then-Filter module
    """
    device = args.device
    
    if args.using_image_features:
        with open(inpath, 'rb') as infile:
            annotations = pickle.load(infile) # [[image_path, image_features, [caption1, caption2, ...]], ...]
    else:
        with open(inpath, 'r') as infile:
            annotations = json.load(infile)   # {image_path: [caption1, caption2, ...]}

    predicts = []
    for idx, item in tqdm(enumerate(annotations)):
        if args.using_image_features:
            image_id, image_features, captions = item
            image_features = image_features.float().unsqueeze(dim = 0).to(device) # (1, clip_hidden_size)
        else:
            image_id = item
            captions = annotations[item]
            image_path = os.path.join(args.image_folder, image_id)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue
            image = preprocess(Image.open(image_path)).unsqueeze(dim = 0).to(device)
            image_features = encoder.encode_image(image).float()
        
        image_features /= image_features.norm(2, dim = -1, keepdim = True)
        continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
        
        if args.using_hard_prompt:
            # MeaCap Retrieve-then-Filter module
            if args.using_image_features:
                # If using pre-extracted features, we need to compute image embeddings for retrieval
                # Note: This is a simplified approach. For better accuracy, you might want to 
                # pre-compute image embeddings for the validation set
                image_path = os.path.join(args.image_folder, image_id)
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}, skipping...")
                    continue
                batch_image_embeds = vl_model.compute_image_representation_from_image_path(image_path)
            else:
                # image_path already computed above
                batch_image_embeds = vl_model.compute_image_representation_from_image_path(image_path)

            # Retrieve: Find top-K similar memory captions
            if retrieve_on_CPU:
                batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
                clip_score_cpu, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                    batch_image_embeds_cpu, memory_clip_embeddings
                )
                clip_score = clip_score_cpu.to(device)
            else:
                clip_score, _ = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                    batch_image_embeds, memory_clip_embeddings
                )
            
            select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
            select_memory_captions = [memory_captions[id] for id in select_memory_ids]

            # Filter: Extract key concepts using Retrieve-then-Filter
            detected_objects = retrieve_concepts(
                parser_model=parser_model,
                parser_tokenizer=parser_tokenizer,
                wte_model=wte_model,
                select_memory_captions=select_memory_captions,
                image_embeds=batch_image_embeds,
                device=device
            )

            # Compose discrete prompts (same as original ViECap)
            discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)
            discrete_embeddings = model.word_embed(discrete_tokens)
            
            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
        else:
            embeddings = continuous_embeddings
        
        # Text generation (same as original ViECap)
        if 'gpt' in args.language_model.lower():
            if not args.using_greedy_search:
                sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
                sentence = sentence[0] # selected top 1
            else:
                sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
        else:
            sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
            sentence=sentence[0]
        
        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
    
    # Save results with _meacap suffix to distinguish from original
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions_meacap.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)
    
    print(f"Saved {len(predicts)} predictions to {out_json_path}")


@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # Loading ViECap model
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    model.to(device)
    
    # Loading CLIP encoder for image encoding (ViECap's soft prompt)
    if not args.using_image_features:
        encoder, preprocess = clip.load(args.clip_model, device = device)
    else:
        encoder, preprocess = None, None
    
    # Loading MeaCap modules
    print("Loading MeaCap modules...")
    
    # 1. CLIP model for retrieval
    if os.path.exists(args.vl_model):
        vl_model_path = args.vl_model
    elif os.path.exists(os.path.join('checkpoints', args.vl_model.split('/')[-1])):
        vl_model_path = os.path.join('checkpoints', args.vl_model.split('/')[-1])
    else:
        vl_model_path = args.vl_model
    
    vl_model = CLIP(vl_model_path)
    vl_model = vl_model.to(device)
    print(f'Loaded CLIP for retrieval from: {vl_model_path}')

    # 2. SentenceBERT model
    if os.path.exists(args.wte_model_path):
        wte_model_path = args.wte_model_path
    elif os.path.exists(os.path.join('checkpoints', args.wte_model_path.split('/')[-1])):
        wte_model_path = os.path.join('checkpoints', args.wte_model_path.split('/')[-1])
    else:
        wte_model_path = args.wte_model_path
    
    wte_model = SentenceTransformer(wte_model_path)
    print(f'Loaded SentenceBERT from: {wte_model_path}')

    # 3. Flan-T5 parser model
    if os.path.exists(args.parser_checkpoint):
        parser_checkpoint_path = args.parser_checkpoint
    elif os.path.exists(os.path.join('checkpoints', args.parser_checkpoint.split('/')[-1])):
        parser_checkpoint_path = os.path.join('checkpoints', args.parser_checkpoint.split('/')[-1])
    else:
        parser_checkpoint_path = args.parser_checkpoint
    
    parser_tokenizer = AutoTokenizer.from_pretrained(parser_checkpoint_path, local_files_only=args.offline_mode)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(parser_checkpoint_path, local_files_only=args.offline_mode)
    parser_model = parser_model.to(device)
    print(f'Loaded Flan-T5 parser from: {parser_checkpoint_path}')

    # 4. Load memory bank
    memory_id = args.memory_id
    memory_caption_path = args.memory_caption_path
    memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
    memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")
    
    if not os.path.exists(memory_caption_path):
        raise FileNotFoundError(f"Memory caption file not found: {memory_caption_path}")
    if not os.path.exists(memory_clip_embedding_file):
        raise FileNotFoundError(f"Memory CLIP embeddings not found: {memory_clip_embedding_file}")
    if not os.path.exists(memory_wte_embedding_file):
        raise FileNotFoundError(f"Memory SentenceBERT embeddings not found: {memory_wte_embedding_file}")
    
    memory_clip_embeddings = torch.load(memory_clip_embedding_file)
    memory_wte_embeddings = torch.load(memory_wte_embedding_file)
    with open(memory_caption_path, 'r') as f:
        memory_captions = json.load(f)
    
    print(f'Loaded memory bank: {len(memory_captions)} captions from {memory_id}')

    # Handle large memory banks (CC3M/SS1M) on CPU
    if memory_id == 'cc3m' or memory_id == 'ss1m':
        retrieve_on_CPU = True
        print('CC3M/SS1M Memory is too big, using CPU for retrieval...')
        vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
        memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
    else:
        retrieve_on_CPU = False
        vl_model_retrieve = vl_model
        if not memory_clip_embeddings.is_cuda:
            memory_clip_embeddings = memory_clip_embeddings.to(device)
        if not memory_wte_embeddings.is_cuda:
            memory_wte_embeddings = memory_wte_embeddings.to(device)

    # Determine input path
    if args.using_image_features:
        inpath = args.path_of_val_datasets[:-5] + f'_{clip_name}.pickle' # file with image features
    else:
        inpath = args.path_of_val_datasets

    # Run validation
    if args.name_of_datasets == 'nocaps':
        print("NoCaps validation not implemented in this version. Please use original validation.py")
        return
    else:  # coco, flickr30k
        if args.using_image_features:
            validation_coco_flickr30k_meacap(
                args, inpath, model, tokenizer,
                vl_model, vl_model_retrieve,
                memory_captions, memory_clip_embeddings, memory_wte_embeddings,
                parser_model, parser_tokenizer, wte_model, retrieve_on_CPU
            )
        else:
            validation_coco_flickr30k_meacap(
                args, inpath, model, tokenizer,
                vl_model, vl_model_retrieve,
                memory_captions, memory_clip_embeddings, memory_wte_embeddings,
                parser_model, parser_tokenizer, wte_model, retrieve_on_CPU,
                preprocess, encoder
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViECap validation with MeaCap Retrieve-then-Filter module')
    
    # ViECap original arguments
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'gpt2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument('--using_image_features', action = 'store_true', default = False, help = 'using pre-extracted image features')
    parser.add_argument('--name_of_datasets', default = 'coco', choices = ('coco', 'flickr30k', 'nocaps'))
    parser.add_argument('--path_of_val_datasets', default = './annotations/coco/test_captions.json')
    parser.add_argument('--weight_path', default = './checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_folder', default = './annotations/coco/val2014/')
    parser.add_argument('--out_path', default = './checkpoints/train_coco')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--text_prompt', type = str, default = None)
    
    # MeaCap specific arguments
    parser.add_argument('--vl_model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model for retrieval')
    parser.add_argument("--parser_checkpoint", type=str, default='lizhuang144/flan-t5-base-VG-factual-sg', help='Flan-T5 parser checkpoint')
    parser.add_argument("--wte_model_path", type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='SentenceBERT model path')
    parser.add_argument("--memory_id", type=str, default="coco", help="memory bank ID")
    parser.add_argument("--memory_caption_path", type=str, default='data/memory/coco/memory_captions.json', help='memory bank captions file')
    parser.add_argument("--memory_caption_num", type=int, default=5, help='number of memory captions to retrieve')
    parser.add_argument("--offline_mode", action='store_true', default=False, help='Use offline mode (local_files_only=True)')
    
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)
