from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    DenseNet201_Weights,
    EfficientNet_V2_S_Weights,
    RegNet_Y_800MF_Weights,
    ViT_B_16_Weights,
    VGG16_Weights,
    VGG19_Weights,
    efficientnet_v2_s,
    regnet_y_800mf,
    vit_b_16,
)

try:
    import timm
except ImportError:
    timm = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KL_GRADES_5 = {
    0: "Grade 0 — Normal",
    1: "Grade 1 — Doubtful",
    2: "Grade 2 — Mild",
    3: "Grade 3 — Moderate",
    4: "Grade 4 — Severe",
}

KL_GRADES_4 = {
    0: "Grade 0–1 — Normal / Doubtful (merged)",
    1: "Grade 2 — Mild",
    2: "Grade 3 — Moderate",
    3: "Grade 4 — Severe",
}


FUSION_MODEL_ID = "Fusion (MHA)"
ENSEMBLE_MODEL_ID = "Ensemble (Stacking LR)"

BACKBONE_DISPLAY: Dict[str, str] = {
    "VGG16": "VGG-16",
    "VGG19": "VGG-19",
    "DenseNet121": "DenseNet-121",
    "DenseNet201": "DenseNet-201",
    "SwinTiny": "Swin Transformer Tiny",
    "ConvNeXt": "ConvNeXt-Tiny",
    "EfficientNetV2": "EfficientNet-V2-S",
    "ViT_B16": "ViT-Base /16",
    "RegNetY008": "RegNetY-008",
}

BACKBONE_FAMILY: Dict[str, str] = {
    "VGG16": "CNN",
    "VGG19": "CNN",
    "DenseNet121": "CNN",
    "DenseNet201": "CNN",
    "SwinTiny": "Transformer",
    "ConvNeXt": "CNN",
    "EfficientNetV2": "CNN",
    "ViT_B16": "Transformer",
    "RegNetY008": "CNN",
}


@dataclass(frozen=True)
class GuiModelOption:
    model_id: str
    label: str
    subtitle: str
    test_acc: Optional[float]
    supports_gradcam: bool
    is_recommended: bool


@dataclass(frozen=True)
class GuiExperimentOption:
    experiment_id: str
    label: str
    subtitle: str
    num_classes: int
    step: int

WEIGHT_STEM = {
    "VGG16": "vgg16",
    "VGG19": "vgg19",
    "DenseNet121": "densenet121",
    "DenseNet201": "densenet201",
    "SwinTiny": "swintiny",
    "ConvNeXt": "convnext",
    "EfficientNetV2": "efficientnetv2",
    "ViT_B16": "vit_b16",
    "RegNetY008": "regnety008",
}

CORE_MODELS = ["VGG16", "VGG19", "DenseNet121", "DenseNet201", "SwinTiny", "ConvNeXt"]
EXTENDED_MODELS = CORE_MODELS + ["EfficientNetV2", "ViT_B16", "RegNetY008"]


def _head(num_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.6),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )


def create_vgg16(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[6] = _head(model.classifier[6].in_features, num_classes)
    return model


def create_vgg19(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[6] = _head(model.classifier[6].in_features, num_classes)
    return model


def create_densenet121(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = _head(model.classifier.in_features, num_classes)
    return model


def create_densenet201(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = _head(model.classifier.in_features, num_classes)
    return model


def create_swin_tiny(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    if timm is None:
        raise ImportError("timm is required for SwinTiny — pip install timm")
    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)


def create_convnext(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = _head(model.classifier[2].in_features, num_classes)
    return model


def create_efficientnetv2(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = _head(model.classifier[1].in_features, num_classes)
    return model


def create_vit_b16(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    model.heads.head = _head(model.heads.head.in_features, num_classes)
    return model


def create_regnety008(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    model = regnet_y_800mf(weights=RegNet_Y_800MF_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = _head(model.fc.in_features, num_classes)
    return model


MODEL_FACTORIES: Dict[str, Callable[..., nn.Module]] = {
    "VGG16": create_vgg16,
    "VGG19": create_vgg19,
    "DenseNet121": create_densenet121,
    "DenseNet201": create_densenet201,
    "SwinTiny": create_swin_tiny,
    "ConvNeXt": create_convnext,
    "EfficientNetV2": create_efficientnetv2,
    "ViT_B16": create_vit_b16,
    "RegNetY008": create_regnety008,
}


class MultiModelAttentionFusion(nn.Module):
    def __init__(
        self,
        backbone_list: List[nn.Module],
        embed_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 5,
        dropout: float = 0.3,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.backbones = nn.ModuleList(backbone_list)
        if freeze_backbones:
            for backbone in self.backbones:
                for param in backbone.parameters():
                    param.requires_grad = False

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(bb.out_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                )
                for bb in backbone_list
            ]
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, len(backbone_list) + 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for backbone, projector in zip(self.backbones, self.projectors):
            with torch.no_grad():
                feat = backbone(x)
            feats.append(projector(feat))
        tokens = torch.stack(feats, dim=1)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1) + self.pos_embedding
        attended = self.transformer(tokens)
        return self.classifier(attended[:, 0])


class _VGGFeatures(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        base = models.vgg16(weights=None) if variant == "vgg16" else models.vgg19(weights=None)
        base.classifier = base.classifier[:-1]
        self.model = base
        self.out_dim = 4096

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _DenseNetFeatures(nn.Module):
    def __init__(self, variant: str):
        super().__init__()
        base = models.densenet121(weights=None) if variant == "densenet121" else models.densenet201(weights=None)
        self.out_dim = base.classifier.in_features
        base.classifier = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _SwinTinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for SwinTiny")
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        self.out_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _ConvNeXtFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.out_dim = 768
        base.classifier[2] = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _EfficientNetV2Features(nn.Module):
    def __init__(self):
        super().__init__()
        base = efficientnet_v2_s(weights=None)
        self.out_dim = base.classifier[1].in_features
        base.classifier = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _ViTFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        base = vit_b_16(weights=None)
        self.out_dim = base.heads.head.in_features
        base.heads.head = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class _RegNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        base = regnet_y_800mf(weights=None)
        self.out_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


FEATURE_EXTRACTORS: Dict[str, Callable[[], nn.Module]] = {
    "VGG16": lambda: _VGGFeatures("vgg16"),
    "VGG19": lambda: _VGGFeatures("vgg19"),
    "DenseNet121": lambda: _DenseNetFeatures("densenet121"),
    "DenseNet201": lambda: _DenseNetFeatures("densenet201"),
    "SwinTiny": _SwinTinyFeatures,
    "ConvNeXt": _ConvNeXtFeatures,
    "EfficientNetV2": _EfficientNetV2Features,
    "ViT_B16": _ViTFeatures,
    "RegNetY008": _RegNetFeatures,
}


EXPERIMENT_PROFILES: Dict[str, dict] = {
    "step10_4class_es": {
        "folder": os.path.join(BASE_DIR, "4 Classes early stop"),
        "notebook": "4 classes early stop.ipynb",
        "step": 10,
        "num_classes": 4,
        "early_stop": True,
        "weight_suffix": "_4class",
        "ensemble_file": "ensemble_full_checkpoint_4class.pth",
        "fusion_file": "best_fusion_multihead_attention_4class.pth",
        "models": EXTENDED_MODELS,
        "gui_title": "Step 10 · 4-Class + Early Stop (Recommended)",
        "gui_subtitle": "Best accuracy · Stacking LR 83.15% · 9 backbones · GAN data",
        "best_method": "Stacking (LR) 83.15%",
        "default_model_id": ENSEMBLE_MODEL_ID,
    },
    "step8_4class": {
        "folder": os.path.join(BASE_DIR, "4 Classes"),
        "notebook": "4Classes.ipynb",
        "step": 8,
        "num_classes": 4,
        "early_stop": False,
        "weight_suffix": "_4class",
        "ensemble_file": "ensemble_full_checkpoint_4class.pth",
        "fusion_file": "best_fusion_multihead_attention_4class.pth",
        "models": CORE_MODELS,
        "gui_title": "Step 8 · 4-Class (GAN, no early stop)",
        "gui_subtitle": "Grades 0+1 merged · Hard voting 82.67% · 6 backbones",
        "best_method": "Hard Voting 82.67%",
        "default_model_id": ENSEMBLE_MODEL_ID,
    },
    "step9_5class_es": {
        "folder": os.path.join(BASE_DIR, "5 classes early stop"),
        "notebook": "5 classes early stop.ipynb",
        "step": 9,
        "num_classes": 5,
        "early_stop": True,
        "weight_suffix": "",
        "ensemble_file": "ensemble_full_checkpoint.pth",
        "fusion_file": "best_fusion_multihead_attention.pth",
        "models": EXTENDED_MODELS,
        "gui_title": "Step 9 · 5-Class + Early Stop (GAN)",
        "gui_subtitle": "Full KL 0–4 · Hard Voting 72.10% · 9 backbones",
        "best_method": "Hard Voting 72.10%",
        "default_model_id": ENSEMBLE_MODEL_ID,
    },
    "step7_5class": {
        "folder": os.path.join(BASE_DIR, "5 Classes"),
        "notebook": "5 Classes.ipynb",
        "step": 7,
        "num_classes": 5,
        "early_stop": False,
        "weight_suffix": "",
        "ensemble_file": "ensemble_full_checkpoint.pth",
        "fusion_file": "best_fusion_multihead_attention.pth",
        "models": CORE_MODELS,
        "gui_title": "Step 7 · 5-Class (GAN, no early stop)",
        "gui_subtitle": "Full KL 0–4 · Hard voting 72.40% · 6 backbones",
        "best_method": "Hard Voting 72.40%",
        "default_model_id": ENSEMBLE_MODEL_ID,
    },
}

DEFAULT_EXPERIMENT_ID = "step10_4class_es"
EXPERIMENT_MENU_ORDER = [
    "step10_4class_es",
    "step8_4class",
    "step9_5class_es",
    "step7_5class",
]


def get_class_labels(num_classes: int) -> Dict[int, str]:
    return KL_GRADES_5 if num_classes == 5 else KL_GRADES_4


def weight_path(profile: dict, model_name: str) -> str:
    stem = WEIGHT_STEM[model_name]
    return os.path.join(profile["folder"], f"best_{stem}{profile['weight_suffix']}.pth")


def fusion_path(profile: dict) -> str:
    return os.path.join(profile["folder"], profile["fusion_file"])


def ensemble_path(profile: dict) -> str:
    return os.path.join(profile["folder"], profile["ensemble_file"])


def list_available_models(profile: dict) -> List[str]:
    return [opt.model_id for opt in get_gui_model_options(profile)]


def load_test_accuracy_table(profile: dict) -> Dict[str, float]:
    csv_path = os.path.join(profile["folder"], "model_comparison_results.csv")
    if not os.path.isfile(csv_path):
        return {}

    table: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Model", "").strip()
            if not name:
                continue
            try:
                acc = float(row.get("Test Acc", 0))
            except ValueError:
                continue
            if name in table:
                table[name] = max(table[name], acc)
            else:
                table[name] = acc
    return table


def get_best_ensemble_from_checkpoint(profile: dict) -> Tuple[Optional[str], Optional[float]]:
    path = ensemble_path(profile)
    if not os.path.isfile(path):
        return None, None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        best = ckpt.get("best_ensemble")
        accs = ckpt.get("ensemble_accuracies", {})
        if best and best in accs:
            return str(best), float(accs[best])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None, None


def get_experiment_subtitle(profile: dict) -> str:
    best_name, best_acc = get_best_ensemble_from_checkpoint(profile)
    if best_name is not None and best_acc is not None:
        class_txt = "Full KL 0–4" if profile["num_classes"] == 5 else "Grades 0+1 merged"
        es_note = " · early stop" if profile.get("early_stop") else ""
        return (
            f"{class_txt}{es_note} · {best_name} {best_acc:.2f}% · "
            f"{len(profile['models'])} backbones"
        )
    return profile["gui_subtitle"]


def get_ensemble_stacking_accuracy(profile: dict) -> Optional[float]:
    path = ensemble_path(profile)
    if not os.path.isfile(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        accs = ckpt.get("ensemble_accuracies", {})
        best = ckpt.get("best_ensemble", "Stacking (LR)")
        if "Stacking (LR)" in accs:
            return float(accs["Stacking (LR)"])
        if best in accs:
            return float(accs[best])
        return None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def format_backbone_label(model_id: str, test_acc: Optional[float]) -> str:
    name = BACKBONE_DISPLAY.get(model_id, model_id)
    family = BACKBONE_FAMILY.get(model_id, "Model")
    if test_acc is not None:
        return f"{name} ({family}) — {test_acc:.2f}% test"
    return f"{name} ({family})"


def get_gui_experiment_options() -> List[GuiExperimentOption]:
    options: List[GuiExperimentOption] = []
    for exp_id in EXPERIMENT_MENU_ORDER:
        profile = EXPERIMENT_PROFILES[exp_id]
        options.append(
            GuiExperimentOption(
                experiment_id=exp_id,
                label=profile["gui_title"],
                subtitle=get_experiment_subtitle(profile),
                num_classes=profile["num_classes"],
                step=profile["step"],
            )
        )
    return options


def get_gui_model_options(profile: dict) -> List[GuiModelOption]:
    acc_table = load_test_accuracy_table(profile)
    options: List[GuiModelOption] = []

    stacking_acc = get_ensemble_stacking_accuracy(profile)
    if stacking_acc is None:
        stacking_acc = acc_table.get("Stacking (LR)")
    if os.path.isfile(ensemble_path(profile)):
        options.append(
            GuiModelOption(
                model_id=ENSEMBLE_MODEL_ID,
                label="⭐ Stacking Ensemble (Logistic Regression)",
                subtitle=(
                    f"Combines all {len(profile['models'])} backbones · "
                    f"{stacking_acc:.2f}% test · weighted Grad-CAM"
                    if stacking_acc is not None
                    else "Combines all trained backbones · weighted Grad-CAM"
                ),
                test_acc=stacking_acc,
                supports_gradcam=True,
                is_recommended=True,
            )
        )

    fusion_acc = acc_table.get("Fusion (MHA)")
    if os.path.isfile(fusion_path(profile)):
        top3 = get_fusion_top3(profile)
        top3_names = ", ".join(BACKBONE_DISPLAY.get(m, m) for m in top3)
        options.append(
            GuiModelOption(
                model_id=FUSION_MODEL_ID,
                label="⭐ Multi-Head Attention Fusion (MHA)",
                subtitle=(
                    f"Top-3: {top3_names} · {fusion_acc:.2f}% test · fused Grad-CAM"
                    if fusion_acc is not None
                    else f"Top-3 backbones: {top3_names} · fused Grad-CAM"
                ),
                test_acc=fusion_acc,
                supports_gradcam=True,
                is_recommended=True,
            )
        )

    backbones: List[GuiModelOption] = []
    for model_id in profile["models"]:
        if not os.path.isfile(weight_path(profile, model_id)):
            continue
        acc = acc_table.get(model_id)
        backbones.append(
            GuiModelOption(
                model_id=model_id,
                label=format_backbone_label(model_id, acc),
                subtitle=f"{BACKBONE_FAMILY.get(model_id, 'Model')} backbone · Grad-CAM supported",
                test_acc=acc,
                supports_gradcam=True,
                is_recommended=False,
            )
        )
    backbones.sort(key=lambda o: (o.test_acc is not None, o.test_acc or 0), reverse=True)
    options.extend(backbones)
    return options


def model_supports_gradcam(model_id: str) -> bool:
    return True


def get_model_display_name(model_id: str, profile: Optional[dict] = None) -> str:
    if model_id == ENSEMBLE_MODEL_ID:
        return "Stacking Ensemble (Logistic Regression)"
    if model_id == FUSION_MODEL_ID:
        if profile:
            top3 = get_fusion_top3(profile)
            names = ", ".join(BACKBONE_DISPLAY.get(m, m) for m in top3)
            return f"Multi-Head Attention Fusion ({names})"
        return "Multi-Head Attention Fusion (MHA)"
    return format_backbone_label(model_id, None)


def _load_state_dict(path: str, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def get_fusion_top3(profile: dict) -> List[str]:
    csv_path = os.path.join(profile["folder"], "model_comparison_results.csv")
    if not os.path.isfile(csv_path):
        return profile["models"][:3]

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Model", "").strip()
            if not name or "fusion" in name.lower():
                continue
            try:
                acc = float(row.get("Test Acc", 0))
            except ValueError:
                continue
            if name in profile["models"]:
                existing = next((i for i, (n, _) in enumerate(rows) if n == name), None)
                if existing is not None:
                    if acc > rows[existing][1]:
                        rows[existing] = (name, acc)
                else:
                    rows.append((name, acc))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in rows[:3]]


def load_individual_model(
    profile: dict, model_name: str, device: torch.device
) -> Tuple[nn.Module, str]:
    path = weight_path(profile, model_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Weight file not found: {path}")

    factory = MODEL_FACTORIES[model_name]
    model = factory(num_classes=profile["num_classes"], pretrained=False)
    model.load_state_dict(_load_state_dict(path, device), strict=True)
    model.to(device).eval()
    return model, model_name


def load_fusion_model(profile: dict, device: torch.device) -> Tuple[nn.Module, str]:
    path = fusion_path(profile)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fusion weights not found: {path}")

    top3 = get_fusion_top3(profile)
    backbones = [FEATURE_EXTRACTORS[name]() for name in top3]
    fusion = MultiModelAttentionFusion(
        backbone_list=backbones,
        embed_dim=512,
        num_heads=8,
        num_classes=profile["num_classes"],
        dropout=0.3,
        freeze_backbones=True,
    )
    fusion.load_state_dict(_load_state_dict(path, device), strict=True)
    fusion.to(device).eval()
    top3_display = ", ".join(BACKBONE_DISPLAY.get(m, m) for m in top3)
    return fusion, f"Multi-Head Attention Fusion ({top3_display})"


def _build_fusion_from_state(
    profile: dict, fusion_state: dict, device: torch.device
) -> MultiModelAttentionFusion:
    top3 = get_fusion_top3(profile)
    backbones = [FEATURE_EXTRACTORS[name]() for name in top3]
    fusion = MultiModelAttentionFusion(
        backbone_list=backbones,
        embed_dim=512,
        num_heads=8,
        num_classes=profile["num_classes"],
        dropout=0.3,
        freeze_backbones=True,
    )
    fusion.load_state_dict(fusion_state, strict=True)
    fusion.to(device).eval()
    return fusion


class EnsembleStackingPredictor:
    def __init__(self, profile: dict, device: torch.device):
        path = ensemble_path(profile)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Ensemble checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.device = device
        self.profile = profile
        self.model_names: List[str] = list(ckpt["ensemble_config"]["models"])
        self.num_classes: int = int(ckpt["ensemble_config"]["num_classes"])
        self.meta = ckpt["stacking_meta_learner"]
        self.models: Dict[str, nn.Module] = {}
        self.fusion_model: Optional[MultiModelAttentionFusion] = None

        missing = [n for n in self.model_names if n not in ckpt.get("individual_weights", {})]
        if missing:
            raise FileNotFoundError(
                f"Ensemble checkpoint missing weights for: {', '.join(missing)}"
            )

        for name in self.model_names:
            try:
                factory = MODEL_FACTORIES[name]
                model = factory(num_classes=self.num_classes, pretrained=False)
                model.load_state_dict(ckpt["individual_weights"][name], strict=True)
                model.to(device).eval()
                self.models[name] = model
            except ImportError as exc:
                raise ImportError(
                    f"Cannot load backbone '{name}' for ensemble — {exc}"
                ) from exc

        fusion_state = ckpt.get("fusion_weights")
        if fusion_state is not None:
            self.fusion_model = _build_fusion_from_state(profile, fusion_state, device)

        self._validate_meta_feature_dim()

        accs = ckpt.get("ensemble_accuracies", {})
        acc = accs.get("Stacking (LR)")
        acc_txt = f" · {acc:.2f}% test" if isinstance(acc, (int, float)) else ""
        self.display_name = f"Stacking Ensemble (Logistic Regression){acc_txt}"

    def _expected_feature_dim(self) -> int:
        n_sources = len(self.model_names) + (1 if self.fusion_model is not None else 0)
        return n_sources * self.num_classes

    def _validate_meta_feature_dim(self) -> None:
        if not isinstance(self.meta, dict) or "coef" not in self.meta:
            return
        coef = np.asarray(self.meta["coef"], dtype=np.float64)
        expected = self._expected_feature_dim()
        if coef.shape[1] != expected:
            raise ValueError(
                f"Stacking meta-learner expects {coef.shape[1]} features but "
                f"ensemble provides {expected} ({len(self.model_names)} backbones"
                f"{' + fusion' if self.fusion_model else ''}) × {self.num_classes} classes."
            )

    def _collect_stacking_features(self, tensor: torch.Tensor) -> np.ndarray:
        probs_list = []
        with torch.no_grad():
            for name in self.model_names:
                out = self.models[name](tensor)
                if isinstance(out, tuple):
                    out = out[0]
                probs_list.append(torch.softmax(out, dim=1).cpu().numpy())
            if self.fusion_model is not None:
                fout = self.fusion_model(tensor)
                probs_list.append(torch.softmax(fout, dim=1).cpu().numpy())
        return np.concatenate(probs_list, axis=1)

    @staticmethod
    def _softmax_rows(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

    def _meta_predict_proba(self, features: np.ndarray) -> np.ndarray:
        meta = self.meta
        if hasattr(meta, "predict_proba"):
            return np.asarray(meta.predict_proba(features)[0], dtype=np.float64)

        if not isinstance(meta, dict):
            raise TypeError(f"Unsupported stacking meta-learner type: {type(meta)}")

        if "coef" not in meta or "intercept" not in meta:
            raise KeyError("stacking_meta_learner dict must contain 'coef' and 'intercept'")

        coef = np.asarray(meta["coef"], dtype=np.float64)
        intercept = np.asarray(meta["intercept"], dtype=np.float64)
        logits = features @ coef.T + intercept
        proba = self._softmax_rows(logits)[0]

        classes = meta.get("classes")
        if classes is not None:
            full = np.zeros(self.num_classes, dtype=np.float64)
            for i, cls in enumerate(classes):
                idx = int(cls)
                if 0 <= idx < self.num_classes:
                    full[idx] = proba[i]
            if full.sum() > 0:
                return full / full.sum()
        return proba

    def _soft_voting_from_features(self, stacked: np.ndarray) -> np.ndarray:
        n = self.num_classes
        n_sources = stacked.shape[1] // n
        chunks = [stacked[0, i * n : (i + 1) * n] for i in range(n_sources)]
        return np.mean(chunks, axis=0)

    def predict_proba(self, tensor: torch.Tensor) -> np.ndarray:
        stacked = self._collect_stacking_features(tensor)
        try:
            return self._meta_predict_proba(stacked)
        except Exception:
            return self._soft_voting_from_features(stacked)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        proba = self.predict_proba(tensor)
        return torch.tensor(proba, dtype=torch.float32).unsqueeze(0)


def load_model(
    profile: dict, model_name: str, device: torch.device
) -> Tuple[nn.Module, str]:
    if model_name == FUSION_MODEL_ID:
        return load_fusion_model(profile, device)
    if model_name == ENSEMBLE_MODEL_ID:
        predictor = EnsembleStackingPredictor(profile, device)
        return predictor, predictor.display_name
    model, _ = load_individual_model(profile, model_name, device)
    acc = load_test_accuracy_table(profile).get(model_name)
    display = format_backbone_label(model_name, acc)
    return model, display


def _gradcam_architecture_kind(model_name: str) -> str:
    if model_name in ("SwinTiny", "ViT_B16"):
        return "transformer"
    return "cnn"


def _feature_extractor_target_layer(
    backbone: nn.Module,
) -> Tuple[Optional[nn.Module], str]:
    name = type(backbone).__name__
    if "VGG" in name:
        return backbone.model.features[-1], "cnn"
    if "DenseNet" in name:
        return backbone.model.features.denseblock4, "cnn"
    if "ConvNeXt" in name:
        return backbone.model.features[-1], "cnn"
    if "SwinTiny" in name:
        return backbone.model.layers[-1].blocks[-1].norm2, "transformer"
    if "EfficientNet" in name:
        return backbone.model.features[-1], "cnn"
    if "ViT" in name:
        return backbone.model.encoder.layers[-1].ln_1, "transformer"
    if "RegNet" in name:
        return backbone.model.trunk_output[-1], "cnn"
    return None, "cnn"


def _cam_from_activation_gradient(
    act: torch.Tensor, grad: torch.Tensor, kind: str, out_size: int
) -> Optional[np.ndarray]:
    if kind == "cnn" and act.dim() == 4 and grad.dim() == 4:
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    elif kind == "transformer" and act.dim() == 3 and grad.dim() == 3:
        weights = grad.mean(dim=-1, keepdim=True)
        ct = torch.relu((weights * act).sum(dim=-1))
        n_tok = ct.shape[1]
        side = int(np.ceil(np.sqrt(n_tok)))
        padded = torch.zeros(1, side * side, device=ct.device, dtype=ct.dtype)
        padded[:, :n_tok] = ct
        cam = padded.view(1, 1, side, side)
    else:
        return None

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_up = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return cam_up.squeeze().detach().cpu().numpy()


def _single_model_gradcam(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target_class: int,
    out_size: int,
) -> Optional[np.ndarray]:
    target_layer = gradcam_target_layer(model, model_name)
    if target_layer is None:
        return None
    kind = _gradcam_architecture_kind(model_name)
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        activations["v"] = out

    def bwd_hook(_module, _gi, go):
        gradients["v"] = go[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(True)
        input_tensor = input_tensor.detach().requires_grad_(True)
        model.zero_grad()
        with torch.enable_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            output[0, target_class].backward()
        if "v" not in activations or "v" not in gradients:
            return None
        return _cam_from_activation_gradient(
            activations["v"].detach(), gradients["v"].detach(), kind, out_size
        )
    finally:
        fh.remove()
        bh.remove()


def _fusion_forward_with_grad(
    fusion_model: MultiModelAttentionFusion, x: torch.Tensor
) -> torch.Tensor:
    batch_size = x.size(0)
    model_features = []
    for backbone, projector in zip(fusion_model.backbones, fusion_model.projectors):
        model_features.append(projector(backbone(x)))
    tokens = torch.stack(model_features, dim=1)
    cls_tokens = fusion_model.cls_token.expand(batch_size, -1, -1)
    tokens = torch.cat([cls_tokens, tokens], dim=1) + fusion_model.pos_embedding
    attended = fusion_model.transformer(tokens)
    return fusion_model.classifier(attended[:, 0])


def generate_fusion_gradcam(
    fusion_model: MultiModelAttentionFusion,
    input_tensor: torch.Tensor,
    target_class: int,
    out_size: int = 224,
) -> Optional[np.ndarray]:
    backbone_activations: Dict[int, torch.Tensor] = {}
    backbone_gradients: Dict[int, torch.Tensor] = {}
    backbone_kinds: Dict[int, str] = {}
    hooks = []

    for idx, backbone in enumerate(fusion_model.backbones):
        layer, kind = _feature_extractor_target_layer(backbone)
        if layer is None:
            continue
        backbone_kinds[idx] = kind
        fh = layer.register_forward_hook(
            lambda _m, _inp, out, i=idx: backbone_activations.__setitem__(i, out)
        )
        bh = layer.register_full_backward_hook(
            lambda _m, _gi, go, i=idx: backbone_gradients.__setitem__(i, go[0])
        )
        hooks.extend([fh, bh])

    try:
        fusion_model.eval()
        for backbone in fusion_model.backbones:
            for param in backbone.parameters():
                param.requires_grad_(True)
        input_tensor = input_tensor.detach().requires_grad_(True)
        fusion_model.zero_grad()
        with torch.enable_grad():
            output = _fusion_forward_with_grad(fusion_model, input_tensor)
            output[0, target_class].backward()

        cams: List[np.ndarray] = []
        for idx in sorted(backbone_activations.keys()):
            if idx not in backbone_gradients:
                continue
            cam = _cam_from_activation_gradient(
                backbone_activations[idx].detach(),
                backbone_gradients[idx].detach(),
                backbone_kinds.get(idx, "cnn"),
                out_size,
            )
            if cam is not None:
                cams.append(cam)
        if not cams:
            return None
        avg = np.mean(cams, axis=0)
        return (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
    finally:
        for hook in hooks:
            hook.remove()
        for backbone in fusion_model.backbones:
            for param in backbone.parameters():
                param.requires_grad_(False)


def generate_ensemble_gradcam(
    predictor: EnsembleStackingPredictor,
    input_tensor: torch.Tensor,
    target_class: int,
    out_size: int = 224,
) -> Optional[np.ndarray]:
    acc_table = load_test_accuracy_table(predictor.profile)
    weights: Dict[str, float] = {}
    for name in predictor.model_names:
        acc = acc_table.get(name)
        weights[name] = float(acc) if acc is not None and acc > 0 else 1.0
    total = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}

    weighted_sum = np.zeros((out_size, out_size), dtype=np.float64)
    weight_used = 0.0
    for name in predictor.model_names:
        cam = _single_model_gradcam(
            predictor.models[name], name, input_tensor, target_class, out_size
        )
        if cam is None:
            continue
        w = weights.get(name, 0.0)
        weighted_sum += w * cam
        weight_used += w

    if weight_used <= 0:
        return None
    result = weighted_sum / weight_used
    return (result - result.min()) / (result.max() - result.min() + 1e-8)


def generate_gradcam_heatmap(
    model: nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    target_class: int,
    out_size: int = 224,
) -> Optional[np.ndarray]:
    if isinstance(model, EnsembleStackingPredictor):
        return generate_ensemble_gradcam(model, input_tensor, target_class, out_size)
    if isinstance(model, MultiModelAttentionFusion):
        return generate_fusion_gradcam(model, input_tensor, target_class, out_size)
    return _single_model_gradcam(model, model_name, input_tensor, target_class, out_size)


def gradcam_target_layer(model: nn.Module, model_name: str) -> Optional[nn.Module]:
    if model_name in ("VGG16", "VGG19"):
        return model.features[-1]
    if model_name == "DenseNet121" or model_name == "DenseNet201":
        return model.features.denseblock4
    if model_name == "ConvNeXt":
        return model.features[-1]
    if model_name == "EfficientNetV2":
        return model.features[-1]
    if model_name == "SwinTiny":
        return model.layers[-1].blocks[-1].norm2
    if model_name == "ViT_B16":
        return model.encoder.layers[-1].ln_1
    if model_name == "RegNetY008":
        return model.trunk_output[-1]
    return None
