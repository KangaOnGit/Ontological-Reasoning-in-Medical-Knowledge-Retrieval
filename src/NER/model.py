import logging
from pathlib import Path

import torch
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.NER.base import Span
from src.utils.config import HF_TOKEN

log = logging.getLogger(__name__)


class NERmodel:
    def __init__(
        self,
        model_name: str,
        prompt_path: str = "configs/prompt/span_extraction.jinja",
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.05,
    ):
        log.info(
            "Model=%s | Prompt=%s | max_new_tokens=%d | repetition_penalty=%.2f",
            model_name,
            prompt_path,
            max_new_tokens,
            repetition_penalty,
        )

        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

        log.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=HF_TOKEN,
        )

        log.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        self.device = next(self.model.parameters()).device

        prompt_path = Path(prompt_path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        env = Environment(
            loader=FileSystemLoader(prompt_path.parent),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self.prompt_template = env.get_template(prompt_path.name)

        log.info("Model loaded successfully.")

    def generate(self, text: str) -> str:
        rendered_prompt = self.prompt_template.render(text=text)

        messages = [
            {
                "role": "user",
                "content": rendered_prompt,
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = outputs[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            response,
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def parse_output(output: str) -> list[Span]:
        if not output:
            return []

        results: list[Span] = []

        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("||", maxsplit=4)]

            if len(parts) != 4:
                log.warning("Skipping malformed output line: %s", line)
                continue
            log.info(f"Processing: {line}")
            results.append(
                Span(
                    text=parts[0],
                    type = parts[1]
                    section=parts[2],
                    subsection=parts[3],
                    context=parts[4],
                )
            )

        return results

    def forward(self, ipt: str) -> list[Span]:
        return self.parse_output(self.generate(ipt))