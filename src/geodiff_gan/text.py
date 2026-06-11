from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


COUNTERFACTUAL_PROMPTS = (
    "dense urban blocks with intersecting roads and compact rooftops",
    "open agricultural parcels with regular field boundaries",
    "continuous forest canopy with irregular natural texture",
    "dry exposed terrain with sparse vegetation and rocky texture",
    "coastal water beside a developed shoreline",
)


class TextEncoder(nn.Module, ABC):
    context_dim: int

    @abstractmethod
    def forward(self, prompts: list[str]) -> torch.Tensor:
        raise NotImplementedError

    def alignment_loss(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        return images.new_zeros(())


class HashTextEncoder(TextEncoder):
    """Deterministic dependency-free encoder for tests and ablations only."""

    def __init__(self, context_dim: int = 768, max_tokens: int = 32) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.max_tokens = max_tokens
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    def _token_vector(self, token: str) -> torch.Tensor:
        seed = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(self.context_dim, generator=generator)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        batch = []
        for prompt in prompts:
            tokens = re.findall(r"[a-z0-9]+", prompt.lower())[: self.max_tokens]
            tokens = tokens or ["<null>"]
            vectors = [self._token_vector(token) for token in tokens]
            while len(vectors) < self.max_tokens:
                vectors.append(torch.zeros(self.context_dim))
            batch.append(torch.stack(vectors))
        return torch.stack(batch).to(self._device_anchor.device)


class TransformersTextEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the caption/text extras to use a pretrained text encoder"
            ) from error
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False).eval()
        self.max_tokens = max_tokens
        config = getattr(self.model.config, "text_config", self.model.config)
        self.context_dim = int(
            getattr(config, "hidden_size", getattr(config, "projection_dim", 768))
        )

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(device)
        if hasattr(self.model, "text_model"):
            outputs = self.model.text_model(**encoded)
        else:
            outputs = self.model(**encoded)
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "text_model_output"):
            return outputs.text_model_output.last_hidden_state
        raise RuntimeError("Text model did not expose token-level hidden states")

    def alignment_loss(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        if not hasattr(self.model, "get_image_features") or not hasattr(
            self.model, "get_text_features"
        ):
            return images.new_zeros(())
        device = images.device
        text_inputs = self.processor(
            text=prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(device)
        image_size = int(
            getattr(getattr(self.model.config, "vision_config", None), "image_size", 224)
        )
        pixel_values = torch.nn.functional.interpolate(
            images, size=(image_size, image_size), mode="bicubic", align_corners=False
        )
        image_processor = getattr(self.processor, "image_processor", None)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        mean_tensor = images.new_tensor(mean)[None, :, None, None]
        std_tensor = images.new_tensor(std)[None, :, None, None]
        pixel_values = (pixel_values - mean_tensor) / std_tensor
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        return 1 - (image_features * text_features).sum(dim=-1).mean()


def build_text_encoder(config: dict) -> TextEncoder:
    text_config = config.get("text_encoder", {})
    kind = text_config.get("kind", "siglip")
    if kind == "hash":
        return HashTextEncoder(
            context_dim=text_config.get("context_dim", config["model"].get("context_dim", 768)),
            max_tokens=text_config.get("max_tokens", 32),
        )
    return TransformersTextEncoder(
        model_name=text_config.get("model_name", "google/siglip-base-patch16-224"),
        max_tokens=text_config.get("max_tokens", 64),
    )


@dataclass(frozen=True)
class PromptBatch:
    prompts: list[str]
    kinds: list[str]


def augment_prompts(
    prompts: list[str],
    null_probability: float = 0.4,
    paraphrase_probability: float = 0.2,
    mismatch_probability: float = 0.1,
    return_metadata: bool = False,
) -> list[str] | PromptBatch:
    if not prompts:
        return PromptBatch(prompts, []) if return_metadata else prompts
    values = prompts.copy()
    kinds = ["original"] * len(values)
    random_values = torch.rand(len(values))
    for index in range(len(values)):
        if random_values[index] < null_probability:
            values[index] = ""
            kinds[index] = "null"
        elif random_values[index] < null_probability + paraphrase_probability:
            values[index] = f"Overhead satellite view containing: {values[index]}"
            kinds[index] = "paraphrase"
        elif (
            random_values[index]
            < null_probability + paraphrase_probability + mismatch_probability
        ):
            if len(values) > 1:
                values[index] = prompts[(index + 1) % len(values)]
            else:
                counterfactual_index = int(
                    torch.randint(0, len(COUNTERFACTUAL_PROMPTS), (1,))
                )
                values[index] = COUNTERFACTUAL_PROMPTS[counterfactual_index]
            kinds[index] = "mismatch"
    result = PromptBatch(values, kinds)
    return result if return_metadata else result.prompts
