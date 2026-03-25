# -*- coding: utf-8 -*-
"""
NoCaps validation with MeaCap Retrieve-then-Filter module.
"""

import os
import json
import pickle
import argparse
import copy
import torch
import clip
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from ClipCap import ClipCaptionModel
from utils import compose_discrete_prompts
try:
    from utils.detect_utils import retrieve_concepts
except ImportError:
    retrieve_concepts = None
from utils.entity_filtering_utils import retrieve_concepts_ef
from models.clip_utils import CLIP
from search import beam_search, greedy_search, opt_search

cpu_device = torch.device("cpu")


def _resolve_model_path(input_path: str) -> str:
    if os.path.exists(input_path):
        return input_path
    cached_candidate = os.path.join("checkpoints", input_path.split("/")[-1])
    if os.path.exists(cached_candidate):
        return cached_candidate
    return input_path


def _generate_sentence(args, model, tokenizer, embeddings):
    lm_name = args.language_model.lower()
    if "gpt" in lm_name:
        if not args.using_greedy_search:
            sentence = beam_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                beam_width=args.beam_width,
                model=model.gpt,
            )[0]
        else:
            sentence = greedy_search(
                embeddings=embeddings,
                tokenizer=tokenizer,
                model=model.gpt,
            )
    else:
        sentence = opt_search(
            prompts=args.text_prompt,
            embeddings=embeddings,
            tokenizer=tokenizer,
            beam_width=args.beam_width,
            model=model.gpt,
        )[0]
    return sentence


def validation_nocaps_meacap(
    args,
    inpath,
    model,
    tokenizer,
    vl_model,
    vl_model_retrieve,
    memory_captions,
    memory_clip_embeddings,
    parser_model,
    parser_tokenizer,
    wte_model,
    retrieve_on_cpu,
    preprocess=None,
    encoder=None,
):
    device = args.device
    if args.using_image_features:
        with open(inpath, "rb") as infile:
            annotations = pickle.load(infile)
    else:
        with open(inpath, "r") as infile:
            annotations = json.load(infile)

    indomain = []
    neardomain = []
    outdomain = []
    overall = []

    for annotation in tqdm(annotations):
        if args.using_image_features:
            image_id, split, image_features, captions = annotation
            image_features = image_features.float().unsqueeze(dim=0).to(device)
            image_path = os.path.join(args.image_folder, split, image_id)
            # In feature mode, use pre-extracted CLIP feature for soft prompt
            # and retrieval to avoid hard dependency on local image files.
            batch_image_embeds = image_features
        else:
            image_id = annotation["image_id"]
            split = annotation["split"]
            captions = annotation["caption"]
            image_path = os.path.join(args.image_folder, split, image_id)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue
            image = preprocess(Image.open(image_path)).unsqueeze(dim=0).to(device)
            image_features = encoder.encode_image(image).float()
            batch_image_embeds = vl_model.compute_image_representation_from_image_path(image_path)

        image_features /= image_features.norm(2, dim=-1, keepdim=True)
        continuous_embeddings = model.mapping_network(image_features).view(
            -1, args.continuous_prompt_length, model.gpt_hidden_size
        )

        if args.using_hard_prompt:
            if retrieve_on_cpu:
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
            select_memory_captions = [memory_captions[idx] for idx in select_memory_ids]

            if args.use_entity_filtering:
                detected_objects = retrieve_concepts_ef(
                    select_memory_captions=select_memory_captions,
                    filter_method=args.ef_filter_method,
                    threshold=args.ef_threshold,
                    alpha=args.ef_alpha,
                    max_entities=args.max_num_of_entities,
                )
            else:
                if retrieve_concepts is None:
                    raise ImportError(
                        "retrieve_concepts not available. Use --use_entity_filtering or fix MeaCap utils."
                    )
                detected_objects = retrieve_concepts(
                    parser_model=parser_model,
                    parser_tokenizer=parser_tokenizer,
                    wte_model=wte_model,
                    select_memory_captions=select_memory_captions,
                    image_embeds=batch_image_embeds,
                    device=device,
                )

            discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim=0).to(device)
            discrete_embeddings = model.word_embed(discrete_tokens)

            if args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim=1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim=1)
        else:
            embeddings = continuous_embeddings

        sentence = _generate_sentence(args, model, tokenizer, embeddings)
        predict = {
            "split": split,
            "image_name": image_id,
            "captions": captions,
            "prediction": sentence,
        }
        overall.append(predict)
        if split == "in_domain":
            indomain.append(predict)
        elif split == "near_domain":
            neardomain.append(predict)
        elif split == "out_domain":
            outdomain.append(predict)

    with open(os.path.join(args.out_path, "overall_generated_captions_meacap.json"), "w") as outfile:
        json.dump(overall, outfile, indent=4)
    with open(os.path.join(args.out_path, "indomain_generated_captions_meacap.json"), "w") as outfile:
        json.dump(indomain, outfile, indent=4)
    with open(os.path.join(args.out_path, "neardomain_generated_captions_meacap.json"), "w") as outfile:
        json.dump(neardomain, outfile, indent=4)
    with open(os.path.join(args.out_path, "outdomain_generated_captions_meacap.json"), "w") as outfile:
        json.dump(outdomain, outfile, indent=4)

    print(
        f"Saved predictions: in={len(indomain)}, near={len(neardomain)}, "
        f"out={len(outdomain)}, overall={len(overall)}"
    )


@torch.no_grad()
def main(args):
    device = args.device
    clip_name = args.clip_model.replace("/", "")
    clip_hidden_size = 640 if "RN" in args.clip_model else 512

    language_model_path = _resolve_model_path(args.language_model)
    tokenizer = AutoTokenizer.from_pretrained(language_model_path, local_files_only=args.offline_mode)
    model = ClipCaptionModel(
        args.continuous_prompt_length,
        args.clip_project_length,
        clip_hidden_size,
        gpt_type=args.language_model,
    )
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"), strict=False)
    model.to(device)

    if args.using_image_features:
        preprocess, encoder = None, None
        inpath = args.path_of_val_datasets[:-5] + f"_{clip_name}.pickle"
    else:
        encoder, preprocess = clip.load(args.clip_model, device=device)
        inpath = args.path_of_val_datasets

    vl_model_path = _resolve_model_path(args.vl_model)
    vl_model = CLIP(vl_model_path).to(device)

    wte_model_path = _resolve_model_path(args.wte_model_path)
    wte_model = SentenceTransformer(wte_model_path)

    parser_checkpoint_path = _resolve_model_path(args.parser_checkpoint)
    parser_tokenizer = AutoTokenizer.from_pretrained(parser_checkpoint_path, local_files_only=args.offline_mode)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(parser_checkpoint_path, local_files_only=args.offline_mode).to(device)

    memory_id = args.memory_id
    memory_caption_path = args.memory_caption_path
    memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
    if not os.path.exists(memory_caption_path):
        raise FileNotFoundError(f"Memory caption file not found: {memory_caption_path}")
    if not os.path.exists(memory_clip_embedding_file):
        raise FileNotFoundError(f"Memory CLIP embeddings not found: {memory_clip_embedding_file}")

    with open(memory_caption_path, "r") as f:
        memory_captions = json.load(f)
    memory_clip_embeddings = torch.load(memory_clip_embedding_file)

    if memory_id in ("cc3m", "ss1m"):
        retrieve_on_cpu = True
        vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
        memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
    else:
        retrieve_on_cpu = False
        vl_model_retrieve = vl_model
        if not memory_clip_embeddings.is_cuda:
            memory_clip_embeddings = memory_clip_embeddings.to(device)

    validation_nocaps_meacap(
        args=args,
        inpath=inpath,
        model=model,
        tokenizer=tokenizer,
        vl_model=vl_model,
        vl_model_retrieve=vl_model_retrieve,
        memory_captions=memory_captions,
        memory_clip_embeddings=memory_clip_embeddings,
        parser_model=parser_model,
        parser_tokenizer=parser_tokenizer,
        wte_model=wte_model,
        retrieve_on_cpu=retrieve_on_cpu,
        preprocess=preprocess,
        encoder=encoder,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoCaps validation with MeaCap Retrieve-then-Filter module")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--language_model", default="gpt2")
    parser.add_argument("--continuous_prompt_length", type=int, default=10)
    parser.add_argument("--clip_project_length", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--using_image_features", action="store_true", default=False)
    parser.add_argument("--name_of_datasets", default="nocaps", choices=("nocaps",))
    parser.add_argument("--path_of_val_datasets", default="./annotations/nocaps/nocaps_corpus.json")
    parser.add_argument("--weight_path", default="./checkpoints/train_coco/coco_prefix-0014.pt")
    parser.add_argument("--image_folder", default="./annotations/nocaps/")
    parser.add_argument("--out_path", default="./checkpoints/train_coco")
    parser.add_argument("--using_hard_prompt", action="store_true", default=False)
    parser.add_argument("--soft_prompt_first", action="store_true", default=False)
    parser.add_argument("--only_hard_prompt", action="store_true", default=False)
    parser.add_argument("--using_greedy_search", action="store_true", default=False)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--text_prompt", type=str, default=None)
    parser.add_argument("--vl_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--parser_checkpoint", type=str, default="lizhuang144/flan-t5-base-VG-factual-sg")
    parser.add_argument("--wte_model_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--memory_id", type=str, default="coco")
    parser.add_argument("--memory_caption_path", type=str, default="data/memory/coco/memory_captions.json")
    parser.add_argument("--memory_caption_num", type=int, default=5)
    parser.add_argument("--offline_mode", action="store_true", default=False)
    parser.add_argument(
        "--use_entity_filtering",
        action="store_true",
        default=False,
        help="Use Entity Filtering (EF) instead of parser-based Retrieve-then-Filter for concept extraction",
    )
    parser.add_argument(
        "--ef_filter_method",
        type=str,
        default="threshold",
        choices=["threshold", "normal", "log_normal"],
        help="EF filter method",
    )
    parser.add_argument(
        "--ef_threshold",
        type=int,
        default=1,
        help="Frequency threshold when ef_filter_method=threshold",
    )
    parser.add_argument(
        "--ef_alpha",
        type=float,
        default=1.0,
        help="Alpha for normal/log_normal EF methods",
    )
    parser.add_argument(
        "--max_num_of_entities",
        type=int,
        default=5,
        help="Max entities to pass to discrete prompts after EF or parser path",
    )
    args = parser.parse_args()
    print(f"args: {vars(args)}\n")
    main(args)
