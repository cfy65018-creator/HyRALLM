import os
import math
import time
import random
import json
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup,
)

def set_seed(seed:int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


python_special_token = ['keyword', 'identifier', 'separators', 'operator', '"', 'integer',
                       'comment', 'none', 'escape_sequence']

java_special_token = ['keyword', 'identifier', 'type_identifier', 'separators', 'operator',
                     'decimal_integer_literal', 'void_type', 'string_literal',
                     'decimal_floating_point_literal', 'boolean_type', 'null_literal',
                     'comment', 'hex_integer_literal', 'character_literal']

special_token_mapping = {
    'JCSD': java_special_token,
    'PCSD': python_special_token,
}

def mask_tokens(inputs, tokenizer, mlm_probability):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability).to(inputs.device)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).to(inputs.device), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100


    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)


    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool().to(inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(inputs.device)
    inputs[indices_random] = random_words[indices_random]


    return inputs, labels

def replace_special_token_with_mask(inputs, special_token_id, tokenizer, mlm_probability):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, 0.0).to(inputs.device)
    probability_matrix.masked_fill_(labels.eq(special_token_id).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels

class PairTextDataset(Dataset):
    def __init__(self, dataset_root:str, dataset_name:str, split:str,
                 augmentation_factor:int = 1, augment_code:bool = True, mlm_probability:float = 0.15):

        jsonl_path = os.path.join(dataset_root, dataset_name, f"{split}.jsonl")

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f": {jsonl_path}")

        self.codes = []
        self.texts = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    js = json.loads(line.strip())
                    code = ' '.join(js['code_tokens']).strip()
                    nl = ' '.join(js['docstring_tokens']).strip()

                    self.codes.append(code)
                    self.texts.append(nl)
                except Exception as e:
                    print(f"  Warning: skip line {idx}, parse failed: {e}")
                    continue


        self.augmentation_factor = augmentation_factor if split == 'train' and augment_code else 1
        self.augment_code = augment_code and split == 'train'
        self.original_size = len(self.codes)
        self.split = split
        self.dataset_name = dataset_name
        self.mlm_probability = mlm_probability

        print(f"Dataset {dataset_name}/{split}:")
        print(f"  - Original samples: {self.original_size}")
        print(f"  - Augmentation factor: {self.augmentation_factor}")
        print(f"  - Effective samples: {len(self)}")
        if self.augment_code:
            print(f"  - Code augmentation: DM + DMST")
            print(f"  - Mask probability: {self.mlm_probability}")
        else:
            print(f"  - Code augmentation: disabled")

    def __len__(self) -> int:
        return self.original_size * self.augmentation_factor

    def __getitem__(self, idx:int) -> Tuple[str, str, str, str]:
        original_idx = idx % self.original_size
        augment_version = idx // self.original_size

        code = self.codes[original_idx]
        text = self.texts[original_idx]
        augmentation_type = 'original'


        if augment_version > 0 and self.augment_code:
            if augment_version % 2 == 1:
                augmentation_type = 'dm'
            else:
                augmentation_type = 'dmst'

        return code, text, augmentation_type, self.dataset_name

def collate_fn(examples:List[Tuple[str, str, str, str]], tokenizer, max_source_length:int, max_target_length:int):
    codes, texts, augmentation_types, dataset_names = zip(*examples)
    code_batch = tokenizer(list(codes), padding=True, truncation=True, max_length=max_source_length, return_tensors='pt')
    text_batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_target_length, return_tensors='pt')
    return {
        'input_ids': code_batch['input_ids'],
        'labels': text_batch['input_ids'],
        'augmentation_types': augmentation_types,
        'dataset_names': dataset_names
    }

class ContrastiveGenerator(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_ids(self, input_ids: torch.Tensor, num_beams:int = 4, max_length:int = 64) -> torch.Tensor:
        attn = input_ids.ne(self.tokenizer.pad_token_id)
        return super().generate(
            input_ids=input_ids,
            attention_mask=attn,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
        )

    def sentence_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_like = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_like, p=2, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_sum_loss: bool = False,
        temperature: float = 0.2,
    ):
                if return_sum_loss:
            raise ValueError("This standalone script supports contrastive training only. Set return_sum_loss=False.")

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)


        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc.last_hidden_state


        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        code_feature = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths


        code_feature2 = self.sentence_emb(input_ids, attention_mask)


        nl_feature = self.sentence_emb(labels)


        c2c_scores = torch.einsum("bd,cd->bc", code_feature, code_feature2)


        c2nl_scores = torch.einsum("bd,cd->bc", code_feature, nl_feature)

        batch = input_ids.size(0)
        target = torch.arange(batch, device=input_ids.device)
        ce = nn.CrossEntropyLoss()


        code2code_loss = ce(c2c_scores / temperature, target)
        code2nl_loss = ce(c2nl_scores / temperature, target)


        p_c2c = F.softmax(c2c_scores / temperature, dim=-1)
        p_c2nl = F.softmax(c2nl_scores / temperature, dim=-1)
        align_loss = F.kl_div(p_c2nl.log(), p_c2c.detach(), reduction="batchmean")

        return code2code_loss, code2nl_loss, align_loss

def apply_augmentation(input_ids, augmentation_types, dataset_names, tokenizer, mlm_probability):
    augmented_input_ids = input_ids.clone()

    for i, (aug_type, dataset_name) in enumerate(zip(augmentation_types, dataset_names)):
        if aug_type == 'dm':

            augmented_input_ids[i:i+1], _ = mask_tokens(
                augmented_input_ids[i:i+1], tokenizer, mlm_probability
            )
        elif aug_type == 'dmst':

            special_tokens = special_token_mapping.get(dataset_name.upper(), [])
            special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)


            if special_token_ids:
                selected_token_id = random.choice([tid for tid in special_token_ids if tid != tokenizer.unk_token_id])
                augmented_input_ids[i:i+1], _ = replace_special_token_with_mask(
                    augmented_input_ids[i:i+1], selected_token_id, tokenizer, mlm_probability
                )

    return augmented_input_ids

def train_one_epoch(model, dataloader, optimizer, scheduler, device, log_interval:int, mlm_probability:float):
    model.train()
    running = {
        'c2c': 0.0,
        'c2nl': 0.0,
        'align': 0.0,
        'steps': 0
    }

    augmentation_counts = {
        'original': 0,
        'dm': 0,
        'dmst': 0
    }

    start = time.time()

    for step, batch in enumerate(dataloader, 1):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        augmentation_types = batch['augmentation_types']
        dataset_names = batch['dataset_names']


        for aug_type in augmentation_types:
            if aug_type in augmentation_counts:
                augmentation_counts[aug_type] += 1


        augmented_input_ids = apply_augmentation(
            input_ids, augmentation_types, dataset_names,
            model.tokenizer, mlm_probability
        )

        code2code_loss, code2nl_loss, align_loss = model(
            input_ids=augmented_input_ids,
            labels=labels,
            return_sum_loss=False
        )
        loss = code2code_loss + code2nl_loss + align_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running['c2c'] += float(code2code_loss.detach().item())
        running['c2nl'] += float(code2nl_loss.detach().item())
        running['align'] += float(align_loss.detach().item())
        running['steps'] += 1

        if step % log_interval == 0:
            avg_c2c = running['c2c'] / running['steps']
            avg_c2nl = running['c2nl'] / running['steps']
            avg_align = running['align'] / running['steps']
            elapsed = time.time() - start

            total_samples = sum(augmentation_counts.values())
            dm_ratio = augmentation_counts['dm'] / max(1, total_samples) * 100
            dmst_ratio = augmentation_counts['dmst'] / max(1, total_samples) * 100

            print(f"[train] step={step} avg_c2c={avg_c2c:.4f} avg_c2nl={avg_c2nl:.4f} avg_align={avg_align:.4f} "
                  f"DM:{dm_ratio:.1f}% DMST:{dmst_ratio:.1f}% elapsed={elapsed:.1f}s")

    avg_c2c = running['c2c'] / max(1, running['steps'])
    avg_c2nl = running['c2nl'] / max(1, running['steps'])
    avg_align = running['align'] / max(1, running['steps'])

    total_samples = sum(augmentation_counts.values())
    print(f"\nAugmentation stats for this epoch:")
    print(f"  - Original: {augmentation_counts['original']} ({augmentation_counts['original']/max(1,total_samples)*100:.1f}%)")
    print(f"  - DM: {augmentation_counts['dm']} ({augmentation_counts['dm']/max(1,total_samples)*100:.1f}%)")
    print(f"  - DMST: {augmentation_counts['dmst']} ({augmentation_counts['dmst']/max(1,total_samples)*100:.1f}%)")

    return avg_c2c, avg_c2nl, avg_align, augmentation_counts

def train_contrastive(args):
    set_seed(args.seed)

    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    except Exception as e:
        print(f"[Error] Failed to load tokenizer: {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"  - CLS token: {tokenizer.cls_token} (id: {tokenizer.cls_token_id})")
    print(f"  - SEP token: {tokenizer.sep_token} (id: {tokenizer.sep_token_id})")
    print(f"  - PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"  - Vocab size: {len(tokenizer)}")


    dataset_special_tokens = special_token_mapping.get(args.dataset.upper(), [])
    if dataset_special_tokens:

        new_tokens = [token for token in dataset_special_tokens
                      if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id]
        if new_tokens:
            print(f"Adding new special tokens: {new_tokens}")
            tokenizer.add_tokens(new_tokens)

    try:
        config = T5Config.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
        base = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, local_files_only=args.local_files_only)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        raise

    model = ContrastiveGenerator(config, tokenizer)
    model.load_state_dict(base.state_dict(), strict=False)


    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resize vocab: {model.config.vocab_size} -> {len(tokenizer)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = PairTextDataset(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset,
        split=args.split,
        augmentation_factor=args.augmentation_factor,
        augment_code=args.augment_code,
        mlm_probability=args.mlm_probability
    )

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda ex: collate_fn(ex, tokenizer, args.max_source_length, args.max_target_length),
        drop_last=True
    )

    no_decay = ['bias', 'LayerNorm.weight']
    grouped = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=args.lr)

    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nStart training: epochs={args.epochs} bs={args.batch_size} total_steps={total_steps} warmup={warmup_steps}")
    print(f"Data augmentation strategy: DM + DMST, mask probability: {args.mlm_probability}")
    print(f"Loss functions: code2code + code2nl + alignment (KL divergence)")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEPOCH {epoch}/{args.epochs}")
        print("-" * 40)

        avg_c2c, avg_c2nl, avg_align, augmentation_counts = train_one_epoch(
            model, dl, optimizer, scheduler, device, args.log_interval, args.mlm_probability
        )

        print(f"\n[EPOCH {epoch} complete] avg_c2c={avg_c2c:.4f} avg_c2nl={avg_c2nl:.4f} avg_align={avg_align:.4f}")

    print("\n" + "="*60)
    print("✅ Training complete!")


    if hasattr(args, 'output_dir') and args.output_dir:
        print(f"\n💾 Saving model to: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)


        model_path = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(model.state_dict(), model_path)
        print(f"  - Model state dict saved: {model_path}")


        tokenizer.save_pretrained(args.output_dir)
        print(f"  - Tokenizer saved: {args.output_dir}")


        model.config.save_pretrained(args.output_dir)
        print(f"  - Config saved: {args.output_dir}")

        print("✅ Model saved successfully!")
    else:
        print("⚠️ output_dir is not set, model was not saved to disk")

    return model, tokenizer







