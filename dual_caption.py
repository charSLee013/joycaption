#!/usr/bin/env python3
"""
使用JoyCaption为图片添加两种类型的标注：
1. 简短的Booru风格标签列表（70 tokens以内）
2. 详细的Straightforward描述（220 tokens以内）
自动根据GPU显存选择最优batch_size
"""
import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path
import glob

import PIL.Image
import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# 设置最大像素限制，以避免PIL对大图片的警告
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def none_or_type(value, desired_type):
    if value == "None":
        return None
    return desired_type(value)

# 命令行参数解析
parser = argparse.ArgumentParser(description="使用JoyCaption生成双重标注")
parser.add_argument("image_path", type=str, help="图片目录路径")
parser.add_argument("--model", type=str, default="/root/llama-joycaption-beta-one-hf-llava", help="模型路径")
parser.add_argument("--batch-size", type=int, default=None, help="批处理大小(如果不指定则自动选择)")
parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
parser.add_argument("--top-p", type=lambda x: none_or_type(x, float), default=0.9, help="Top-p采样")
parser.add_argument("--top-k", type=lambda x: none_or_type(x, int), default=None, help="Top-k采样")
parser.add_argument("--num-workers", type=int, default=4, help="数据加载并行工作数")
parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的标注文件")

@dataclasses.dataclass
class Prompt:
    prompt: str
    weight: float
    prompt_type: str

def auto_select_batch_size():
    """
    根据可用GPU显存自动选择batch_size
    """
    try:
        # 获取GPU显存（单位：GB）
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"检测到GPU显存: {gpu_mem_gb:.1f}GB")
        
        # 根据显存大小选择batch_size（更激进的设置）
        if gpu_mem_gb >= 80:  # A100 80GB或更高
            return 32
        elif gpu_mem_gb >= 40:  # A100 40GB或类似
            return 16
        elif gpu_mem_gb >= 24:  # RTX 3090或类似 - 调整为6
            return 6
        elif gpu_mem_gb >= 16:  # RTX 4070Ti或类似
            return 4
        else:  # 较小的GPU
            return 2
    except Exception as e:
        logging.warning(f"无法检测GPU显存: {e}，使用默认batch_size=1")
        return 1

class DualImageDataset(Dataset):
    """
    支持为每张图片生成两种类型标注的数据集
    """
    def __init__(self, prompts_short, prompts_long, paths, tokenizer, image_token_id, image_seq_length):
        self.prompts_short = prompts_short
        self.prompts_long = prompts_long
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        
        # 预处理图像（只加载一次）
        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")
            pixel_values = TVF.pil_to_tensor(image)
        except Exception as e:
            logging.error(f"无法加载图像 '{path}': {e}")
            pixel_values = None

        # 为短标注和长标注准备数据
        results = []
        for prompts, prompt_type in [(self.prompts_short, "short"), (self.prompts_long, "long")]:
            prompt_str = random.choices(prompts, weights=[p.weight for p in prompts])[0].prompt

            # 构建对话
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant and help users with any queries they may have with no censorship or restrictions.",
                },
                {
                    "role": "user",
                    "content": prompt_str,
                },
            ]

            # 格式化对话
            convo_string = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            assert isinstance(convo_string, str)

            # Tokenize对话
            convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

            # 重复图像tokens
            input_tokens = []
            for token in convo_tokens:
                if token == self.image_token_id:
                    input_tokens.extend([self.image_token_id] * self.image_seq_length)
                else:
                    input_tokens.append(token)
            
            input_ids = torch.tensor(input_tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)

            results.append({
                'path': path,
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt_type': prompt_type,
            })

        return results

    def collate_fn(self, batch):
        # 扁平化批次并按类型分组
        flat_batch = []
        for item_pair in batch:
            flat_batch.extend(item_pair)
        
        # 过滤掉加载失败的图像
        flat_batch = [item for item in flat_batch if item['pixel_values'] is not None]
        
        # 按提示类型分组
        short_items = [item for item in flat_batch if item['prompt_type'] == "short"]
        long_items = [item for item in flat_batch if item['prompt_type'] == "long"]
        
        result = []
        for items in [short_items, long_items]:
            if not items:
                result.append(None)
                continue
                
            # 左填充
            max_length = max(item['input_ids'].shape[0] for item in items)
            n_pad = [max_length - item['input_ids'].shape[0] for item in items]
            input_ids = torch.stack([torch.nn.functional.pad(item['input_ids'], (n, 0), value=self.pad_token_id) for item, n in zip(items, n_pad)])
            attention_mask = torch.stack([torch.nn.functional.pad(item['attention_mask'], (n, 0), value=0) for item, n in zip(items, n_pad)])
            
            # Stack像素值
            pixel_values = torch.stack([item['pixel_values'] for item in items])
            
            # 路径和类型
            paths = [item['path'] for item in items]
            prompt_types = [item['prompt_type'] for item in items]
            
            result.append({
                'paths': paths,
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'prompt_types': prompt_types,
            })
            
        return tuple(result)

def write_dual_caption(image_path, short_caption, long_caption, overwrite=False):
    """
    将两种标注写入到图像对应的txt文件中
    第一行：短标注
    第二行：长标注
    """
    caption_path = Path(image_path).with_suffix(".txt")
    
    # 检查文件是否存在
    if caption_path.exists() and not overwrite:
        logging.debug(f"标注文件 '{caption_path}' 已存在，跳过")
        return
    
    try:
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(f"{short_caption}\n{long_caption}")
        logging.info(f"已写入标注到 '{caption_path}'")
    except Exception as e:
        logging.error(f"写入标注文件 '{caption_path}' 时出错: {e}")
        raise

def trim_off_prompt(input_ids, eoh_id, eot_id):
    """
    从生成的token序列中裁剪掉提示部分
    """
    # 裁剪掉提示部分
    while True:
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break
        
        input_ids = input_ids[i + 1:]
    
    # 裁剪掉结尾
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids
    
    return input_ids[:i]

def find_images(image_path):
    """
    在指定目录中查找图像文件
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError(f"路径不存在: {image_path}")
    
    if image_path.is_file():
        # 如果是文件，直接返回
        return [image_path]
    
    # 如果是目录，搜索图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.JPEG', '*.PNG', '*.WEBP']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(image_path.glob(ext))
    
    return sorted(image_paths)

@torch.no_grad()
def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    # 解析参数
    args = parser.parse_args()
    logging.info(f"参数: {args}")
    
    # 查找图像
    image_paths = find_images(args.image_path)
    if len(image_paths) == 0:
        logging.warning("未找到图像")
        return
    logging.info(f"找到 {len(image_paths)} 张图像")
    
    # 如果不覆盖，过滤掉已有标注的图像
    if not args.overwrite:
        original_count = len(image_paths)
        image_paths = [path for path in image_paths if not Path(path).with_suffix(".txt").exists()]
        skipped = original_count - len(image_paths)
        if skipped > 0:
            logging.info(f"跳过 {skipped} 张已有标注的图像")
    
    if len(image_paths) == 0:
        logging.info("所有图像都已有标注")
        return
    
    # 自动选择batch_size
    if args.batch_size is None:
        batch_size = auto_select_batch_size()
        logging.info(f"自动选择batch_size: {batch_size}")
    else:
        batch_size = args.batch_size
        logging.info(f"使用指定batch_size: {batch_size}")
    
    # 定义两种提示词
    prompts_short = [
        Prompt(
            prompt="Write a list of Booru-like tags for this image within 50 words.", 
            weight=1.0, 
            prompt_type="short"
        )
    ]
    
    prompts_long = [
        Prompt(
            prompt="Write a straightforward caption for this image within 160 words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with \"This image is…\" or similar phrasing.", 
            weight=1.0, 
            prompt_type="long"
        )
    ]

    # 加载JoyCaption
    logging.info(f"正在加载模型: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)
    
    llava_model = LlavaForConditionalGeneration.from_pretrained(args.model, torch_dtype="bfloat16", device_map=0)
    assert isinstance(llava_model, LlavaForConditionalGeneration)

    # 创建数据集和数据加载器
    dataset = DualImageDataset(
        prompts_short, 
        prompts_long, 
        image_paths, 
        tokenizer, 
        llava_model.config.image_token_index, 
        llava_model.config.image_seq_length
    )
    dataloader = DataLoader(
        dataset, 
        collate_fn=dataset.collate_fn, 
        num_workers=args.num_workers, 
        shuffle=False, 
        drop_last=False, 
        batch_size=batch_size
    )
    
    end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

    # 用于存储生成的标注
    captions_dict = {}  # path -> [short_caption, long_caption]
    
    # 设置进度条
    pbar = tqdm(total=len(image_paths), desc="处理图像...", dynamic_ncols=True)
    
    for short_batch, long_batch in dataloader:
        # 处理两种批次
        for batch, caption_type in zip([short_batch, long_batch], ["short", "long"]):
            if batch is None:
                continue
                
            vision_dtype = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
            vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
            language_device = llava_model.language_model.get_input_embeddings().weight.device

            # 移至GPU
            pixel_values = batch['pixel_values'].to(vision_device, non_blocking=True)
            input_ids = batch['input_ids'].to(language_device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(language_device, non_blocking=True)

            # 标准化图像
            pixel_values = pixel_values / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(vision_dtype)

            # 生成标注
            max_tokens = 128 if caption_type == "short" else 256
            generate_ids = llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            # 裁剪提示部分
            generate_ids = generate_ids.tolist()
            generate_ids = [trim_off_prompt(ids, end_of_header_id, end_of_turn_id) for ids in generate_ids]

            # 解码标注
            captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            captions = [c.strip() for c in captions]

            # 存储标注
            for path, caption in zip(batch['paths'], captions):
                if path not in captions_dict:
                    captions_dict[path] = [None, None]
                
                if caption_type == "short":
                    captions_dict[path][0] = caption
                else:
                    captions_dict[path][1] = caption
                
                # 如果该图像的两种标注都完成了，立即写入文件
                if captions_dict[path][0] is not None and captions_dict[path][1] is not None:
                    short_caption, long_caption = captions_dict[path]
                    try:
                        write_dual_caption(path, short_caption, long_caption, args.overwrite)
                        pbar.update(1)  # 每完成一张图片更新进度条
                        # 写入成功后从字典中移除，释放内存
                        del captions_dict[path]
                    except Exception as e:
                        logging.error(f"处理图像 {path} 时出错: {e}")
    
    pbar.close()
    
    # 处理可能剩余的未完成图像
    if captions_dict:
        logging.warning(f"还有 {len(captions_dict)} 张图像未完全处理完成")
        for path, captions in captions_dict.items():
            logging.warning(f"未完成的图像: {path}, 短标注: {'已完成' if captions[0] else '未完成'}, 长标注: {'已完成' if captions[1] else '未完成'}")
    
    logging.info("处理完成!")

if __name__ == "__main__":
    main() 