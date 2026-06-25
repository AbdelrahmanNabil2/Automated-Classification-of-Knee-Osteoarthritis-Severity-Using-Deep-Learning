import hashlib
import os
from datetime import datetime
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

from inference_models import (
    DEFAULT_EXPERIMENT_ID,
    ENSEMBLE_MODEL_ID,
    EnsembleStackingPredictor,
    EXPERIMENT_PROFILES,
    FUSION_MODEL_ID,
    generate_gradcam_heatmap,
    get_class_labels,
    get_best_ensemble_from_checkpoint,
    get_gui_experiment_options,
    get_gui_model_options,
    load_model,
    model_supports_gradcam,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass
try:
    torch.set_grad_enabled(False)
except Exception:
    pass


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image)
    if len(image.shape) == 2:
        mn, mx = image.min(), image.max()
        if mx - mn == 0:
            return image.copy()
        return ((image - mn) / (mx - mn) * 255).astype(np.uint8)
    for c in range(image.shape[2]):
        ch = image[:, :, c]
        mn, mx = ch.min(), ch.max()
        if mx - mn == 0:
            result[:, :, c] = ch
        else:
            result[:, :, c] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
    return result


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def sharpen_image(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def process_single_image(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    img = contrast_stretch(image)
    img = histogram_equalization(img)
    img = gamma_correction(img, gamma)
    img = sharpen_image(img)
    return img


@st.cache_resource
def load_cached_model(experiment_id: str, model_id: str):
    profile = EXPERIMENT_PROFILES[experiment_id]
    model, display_name = load_model(profile, model_id, DEVICE)
    return model, display_name, profile


@st.cache_data(show_spinner=False)
def cached_experiment_options():
    return get_gui_experiment_options()


@st.cache_data(show_spinner=False)
def cached_model_options(experiment_id: str):
    return get_gui_model_options(EXPERIMENT_PROFILES[experiment_id])


@st.cache_data(show_spinner=False)
def cached_best_ensemble(experiment_id: str):
    return get_best_ensemble_from_checkpoint(EXPERIMENT_PROFILES[experiment_id])


@st.cache_resource
def _normalize_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def prepare_image(image: np.ndarray, input_size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return _normalize_transform()(rgb).unsqueeze(0)


def predict(model, image, input_size, class_labels):
    tensor = prepare_image(image, input_size).to(DEVICE)
    num_classes = len(class_labels)

    try:
        if isinstance(model, EnsembleStackingPredictor):
            probs = np.asarray(model.predict_proba(tensor), dtype=np.float64)
        else:
            with torch.no_grad():
                outputs = model(tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    except Exception as exc:
        raise RuntimeError(f"Inference failed for {type(model).__name__}: {exc}") from exc

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if probs.size != num_classes:
        raise ValueError(
            f"Model returned {probs.size} probabilities but expected {num_classes} classes."
        )
    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    if total > 0:
        probs = probs / total

    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100.0

    top_k = min(3, len(probs))
    top3_idx = np.argsort(probs)[::-1][:top_k]
    top3 = [(class_labels[int(i)], float(probs[i]) * 100.0) for i in top3_idx]

    return {
        "predicted_class": pred,
        "predicted_label": class_labels[pred],
        "confidence": conf,
        "all_probabilities": probs.tolist(),
        "top3": top3,
    }


def get_gradcam_overlay(image_bgr, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_np = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.6, colormap, 0.4, 0)
    return overlay


def get_gradcam_heatmap_image(image_bgr, heatmap):
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_np = np.uint8(255 * heatmap_resized)
    return cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)


class PatientProfile:
    def __init__(self, age, gender, height_cm, weight_kg, pain_score, walking_diff_score,
                 prev_injury, diabetes, hypertension, activity_level):
        self.age = age
        self.gender = gender
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.pain_score = pain_score
        self.walking_diff_score = walking_diff_score
        self.prev_injury = prev_injury
        self.diabetes = diabetes
        self.hypertension = hypertension
        self.activity_level = activity_level
        self.bmi = self.calculate_bmi()

    def calculate_bmi(self):
        height_m = self.height_cm / 100.0
        if height_m == 0:
            return 0
        return self.weight_kg / (height_m ** 2)


class RiskAssessmentEngine:
    def compute_scores(self, patient: PatientProfile, kl_grade: int):
        progression = kl_grade * 15 + patient.age * 0.2 + patient.bmi * 0.5 + patient.pain_score * 2
        if patient.prev_injury:
            progression += 10
        if patient.diabetes:
            progression += 5
        progression = min(max(progression, 0), 100)

        disability = kl_grade * 10 + patient.walking_diff_score * 5 + patient.pain_score * 3
        if patient.activity_level == "Low":
            disability += 10
        disability = min(max(disability, 0), 100)

        if patient.bmi >= 30:
            obesity_risk = 80 + (patient.bmi - 30) * 2
        elif patient.bmi >= 25:
            obesity_risk = 50 + (patient.bmi - 25) * 2
        else:
            obesity_risk = patient.bmi * 1.5
        obesity_risk = min(max(obesity_risk, 0), 100)

        surgical = kl_grade * 20 + patient.pain_score * 2 + patient.walking_diff_score * 2
        if patient.age > 60:
            surgical += 10
        if kl_grade >= 3:
            surgical += 20
        if patient.bmi > 35:
            surgical += 10
        surgical = min(max(surgical, 0), 100)

        return {
            "Progression Risk Score": self.categorize(progression),
            "Functional Disability Risk": self.categorize(disability),
            "Obesity Risk": self.categorize(obesity_risk),
            "Surgical Need Probability": self.categorize(surgical),
            "_raw": {
                "progression": progression,
                "disability": disability,
                "obesity": obesity_risk,
                "surgical": surgical,
            },
        }

    def categorize(self, score):
        if score <= 30:
            return f"Low ({score:.1f}/100)"
        elif score <= 60:
            return f"Moderate ({score:.1f}/100)"
        else:
            return f"High ({score:.1f}/100)"


def _confidence_tier(confidence: float) -> str:
    if confidence < 50:
        return "very_low"
    if confidence < 65:
        return "borderline"
    if confidence < 80:
        return "moderate"
    return "high"


def _confidence_label(tier: str) -> str:
    return {
        "very_low": "very low",
        "borderline": "borderline",
        "moderate": "moderate",
        "high": "high",
    }[tier]


def _pain_tier(pain: int) -> str:
    if pain <= 3:
        return "mild"
    if pain <= 6:
        return "moderate"
    if pain <= 8:
        return "severe"
    return "very_severe"


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    if bmi < 35:
        return "obese"
    return "severely_obese"


def _prediction_margin(classification: dict) -> float:
    top3 = classification.get("top3", [])
    if len(top3) >= 2:
        return top3[0][1] - top3[1][1]
    return classification.get("confidence", 0.0)


class RecommendationEngine:
    def generate(
        self,
        patient: PatientProfile,
        kl_grade: int,
        risk_scores: dict,
        classification: dict | None = None,
    ):
        classification = classification or {}
        confidence = float(classification.get("confidence", 100.0))
        conf_tier = _confidence_tier(confidence)
        conf_label = _confidence_label(conf_tier)
        margin = _prediction_margin(classification)
        pain_tier = _pain_tier(patient.pain_score)
        bmi_cat = _bmi_category(patient.bmi)
        raw = risk_scores.get("_raw", {})
        surgical_raw = raw.get("surgical", 0)

        recs = {
            "clinical_summary": "",
            "urgency": "Moderate",
            "confidence_tier": conf_tier,
            "confidence_caveat": None,
            "priority_actions": [],
            "Lifestyle Recommendations": [],
            "Exercise Recommendations": [],
            "Pain Management Recommendations": [],
            "Weight Management Recommendations": [],
            "Medical Follow-Up Recommendations": [],
            "Surgical Consultation Recommendations": [],
            "Age-Specific Guidance": [],
            "Clinical Notes": [],
        }

        recs["clinical_summary"] = self._build_summary(
            kl_grade, confidence, conf_label, pain_tier, patient, classification
        )
        recs["confidence_caveat"] = self._confidence_caveat(conf_tier, confidence, margin, classification)
        recs["priority_actions"] = self._priority_actions(
            patient, kl_grade, conf_tier, pain_tier, surgical_raw, margin
        )
        recs["urgency"] = self._urgency_level(patient, kl_grade, pain_tier, surgical_raw, conf_tier)

        recs["Lifestyle Recommendations"] = self._lifestyle_recs(patient, kl_grade, conf_tier, pain_tier)
        recs["Exercise Recommendations"] = self._exercise_recs(patient, kl_grade, conf_tier, bmi_cat)
        recs["Pain Management Recommendations"] = self._pain_recs(patient, kl_grade, pain_tier)
        recs["Weight Management Recommendations"] = self._weight_recs(patient, kl_grade, bmi_cat)
        recs["Medical Follow-Up Recommendations"] = self._followup_recs(
            patient, kl_grade, conf_tier, margin, pain_tier
        )
        recs["Surgical Consultation Recommendations"] = self._surgical_recs(
            patient, kl_grade, conf_tier, surgical_raw, risk_scores
        )
        recs["Age-Specific Guidance"] = self._age_recs(patient, kl_grade, conf_tier, surgical_raw)
        recs["Clinical Notes"] = self._clinical_notes(patient, kl_grade, conf_tier, margin, classification)

        return recs

    def _build_summary(self, kl_grade, confidence, conf_label, pain_tier, patient, classification):
        label = classification.get("predicted_label", f"Grade index {kl_grade}")
        grade_phrase = {
            0: "no radiographic OA",
            1: "doubtful/minimal OA changes",
            2: "mild OA",
            3: "moderate OA with joint space narrowing",
            4: "severe OA",
        }.get(kl_grade, "KOA")

        tier = _confidence_tier(confidence)
        if tier in ("very_low", "borderline"):
            opener = f"KL {label} predicted with {conf_label} confidence ({confidence:.0f}%) — treat as provisional."
        elif tier == "high":
            opener = f"KL {label} confirmed with {conf_label} confidence ({confidence:.0f}%) — {grade_phrase}."
        else:
            opener = f"KL {label} with {conf_label} confidence ({confidence:.0f}%) — {grade_phrase}."

        pain_phrase = {
            "mild": f"Pain {patient.pain_score}/10 (mild)",
            "moderate": f"pain {patient.pain_score}/10 (moderate)",
            "severe": f"pain {patient.pain_score}/10 (severe)",
            "very_severe": f"pain {patient.pain_score}/10 (very severe)",
        }[pain_tier]

        walking_note = ""
        if patient.walking_diff_score >= 7:
            walking_note = " Significant walking difficulty reported."
        elif patient.walking_diff_score >= 4:
            walking_note = " Moderate walking difficulty reported."

        return f"{opener} Patient reports {pain_phrase}.{walking_note}"

    def _confidence_caveat(self, conf_tier, confidence, margin, classification):
        if conf_tier == "very_low":
            return (
                f"CNN confidence is very low ({confidence:.0f}%). Do not commit to an OA-specific pathway yet — "
                "reimage and perform thorough clinical examination."
            )
        if conf_tier == "borderline":
            alt = ""
            top3 = classification.get("top3", [])
            if len(top3) >= 2:
                alt = f" Next most likely grade: {top3[1][0]} ({top3[1][1]:.1f}%)."
            return (
                f"Borderline CNN confidence ({confidence:.0f}%) with a {margin:.1f}% margin over the next grade.{alt} "
                "Correlate with clinical findings before escalating treatment."
            )
        if margin < 10.0 and conf_tier == "moderate":
            return (
                f"Narrow margin ({margin:.1f}%) between top predictions — "
                "clinical correlation recommended before major treatment changes."
            )
        return None

    def _urgency_level(self, patient, kl_grade, pain_tier, surgical_raw, conf_tier):
        if pain_tier == "very_severe" or (patient.pain_score >= 8 and patient.walking_diff_score >= 8):
            return "Urgent"
        if kl_grade >= 4 and conf_tier in ("high", "moderate") and surgical_raw >= 60:
            return "High"
        if pain_tier == "severe" or kl_grade >= 3 or surgical_raw >= 55:
            return "High"
        if kl_grade <= 1 and pain_tier == "mild" and patient.walking_diff_score <= 3:
            return "Low"
        return "Moderate"

    def _priority_actions(self, patient, kl_grade, conf_tier, pain_tier, surgical_raw, margin):
        actions = []

        if conf_tier in ("very_low", "borderline") or margin < 10.0:
            actions.append("Repeat imaging and clinical examination to confirm KL grading before escalation.")

        if pain_tier in ("severe", "very_severe"):
            actions.append("Prioritise stepped analgesia and rapid physiotherapy review within 2 weeks.")
        elif pain_tier == "moderate" and kl_grade >= 2:
            actions.append("Start regular analgesia (topical NSAIDs first) combined with structured exercise.")

        if patient.bmi >= 30:
            actions.append("Refer for structured weight-management support — even 5–10% loss reduces knee load.")

        if kl_grade >= 3 and conf_tier in ("high", "moderate") and surgical_raw >= 55:
            actions.append("Book orthopaedic review to discuss surgical vs. conservative pathways.")
        elif kl_grade >= 4 and conf_tier in ("very_low", "borderline"):
            actions.append("Optimise non-operative care first; do not rush surgical referral at low confidence.")

        if patient.diabetes and kl_grade >= 2:
            actions.append("Coordinate glucose monitoring — uncontrolled diabetes accelerates functional decline.")

        if not actions:
            actions.append("Continue conservative management and reassess symptoms in 8–12 weeks.")

        return actions[:4]

    def _lifestyle_recs(self, patient, kl_grade, conf_tier, pain_tier):
        recs = []

        if kl_grade <= 1:
            recs.append(
                "Continue daily activities with pacing; use cushioned footwear and change position every 30–45 minutes."
            )
        elif kl_grade == 2:
            recs.append(
                "Use activity pacing with rest breaks; avoid prolonged kneeling and deep squatting."
            )
        elif kl_grade == 3:
            recs.append(
                "Modify load-bearing tasks — consider a walking stick, handrails on stairs, and raised seating."
            )
        else:
            recs.append(
                "Prioritise assistive devices, home fall-risk assessment, and energy conservation for daily tasks."
            )

        if patient.activity_level == "Low":
            recs.append(
                "Gradually increase light daily movement (short walks, seated stretches) to counter deconditioning."
            )
        elif patient.activity_level == "High" and kl_grade >= 2:
            recs.append(
                "Temporarily reduce high-impact sport; substitute pool cycling or elliptical training."
            )

        if patient.diabetes:
            recs.append("Maintain glycaemic control — poor diabetes control worsens pain and slows recovery.")
        if patient.hypertension:
            recs.append("Monitor blood pressure regularly; limit prolonged NSAID use without clinician oversight.")

        if patient.prev_injury:
            recs.append("Previous knee injury increases flare risk — avoid sudden load increases after inactivity.")

        return recs

    def _exercise_recs(self, patient, kl_grade, conf_tier, bmi_cat):
        recs = []

        if kl_grade == 0:
            recs.append(
                "Quadriceps strengthening, balance training, and low-impact cardio (cycling, swimming) 3–4×/week."
            )
        elif kl_grade == 1:
            recs.append(
                "Structured programme: VMO activation, mini-squats, hip abductor work, and proprioception 3–4×/week."
            )
        elif kl_grade == 2:
            if conf_tier in ("high", "moderate"):
                recs.append(
                    "Supervised physiotherapy 2–3×/week: progressive resistance, step-ups, and activity pacing."
                )
            else:
                recs.append(
                    "Gentle supervised exercise while awaiting grading confirmation — pool or recumbent cycling preferred."
                )
        elif kl_grade == 3:
            recs.append(
                "Intensive land-based strengthening 3×/week plus aquatic therapy if weight-bearing is limited."
            )
        else:
            recs.append(
                "Preserve mobility with pool or cycle exercise; begin prehabilitation if surgery is being considered."
            )

        if bmi_cat in ("obese", "severely_obese"):
            recs.append("Favour non-weight-bearing options (aquatic therapy, recumbent bike) to protect the joint.")
        if patient.walking_diff_score >= 6:
            recs.append("Include gait retraining and stair-assist strategies with a physiotherapist.")
        if patient.age >= 70:
            recs.append("Emphasise seated and supported exercises to reduce fall risk during rehabilitation.")

        return recs

    def _pain_recs(self, patient, kl_grade, pain_tier):
        recs = []

        if pain_tier == "mild":
            recs.append("Activity modification and heat/cold therapy may suffice; reserve analgesics for flares.")
        elif pain_tier == "moderate":
            recs.append(
                "Stepped analgesia: topical NSAIDs or capsaicin first, then paracetamol; oral NSAIDs only if needed."
            )
        elif pain_tier == "severe":
            recs.append(
                "Prioritise analgesia — topical plus oral NSAIDs with GI protection; consider intra-articular steroid for flares."
            )
        else:
            recs.append(
                "Urgent pain control needed — expedite specialist review and consider joint injection or short opioid course."
            )

        if kl_grade >= 3 and pain_tier in ("moderate", "severe", "very_severe"):
            recs.append("Discuss hyaluronic acid or corticosteroid injection if conservative analgesia is insufficient.")

        if patient.hypertension and pain_tier != "mild":
            recs.append("Use NSAIDs cautiously given hypertension — prefer topical agents and monitor BP.")

        if patient.diabetes and pain_tier != "mild":
            recs.append("Prefer paracetamol and topical agents; monitor renal function if using NSAIDs.")

        return recs

    def _weight_recs(self, patient, kl_grade, bmi_cat):
        if bmi_cat == "normal":
            return ["Maintain current healthy weight to prevent excess joint loading."]

        recs = []
        if bmi_cat == "underweight":
            recs.append("Low BMI may indicate muscle loss — consider protein-rich diet and resistance training.")
            return recs

        if bmi_cat == "overweight":
            recs.append(
                f"BMI {patient.bmi:.1f} (overweight). Target 5% weight reduction over 6 months via dietary counselling."
            )
        elif bmi_cat == "obese":
            recs.append(
                f"BMI {patient.bmi:.1f} (obese). Structured dietitian-led plan — 5–10% loss significantly lowers knee load."
            )
        else:
            recs.append(
                f"BMI {patient.bmi:.1f} (severely obese). Medically supervised weight management is a treatment cornerstone."
            )

        if kl_grade >= 2:
            recs.append("Combine caloric deficit with aquatic or seated exercise to protect the joint during weight loss.")
        if patient.diabetes:
            recs.append("Integrate diabetes education — weight loss improves both glycaemic control and knee symptoms.")

        return recs

    def _followup_recs(self, patient, kl_grade, conf_tier, margin, pain_tier):
        recs = []

        if kl_grade <= 1:
            recs.append("Reassess in 8–12 weeks with KOOS or WOMAC functional scores if symptoms persist.")
        elif kl_grade <= 3:
            interval = "4–6 weeks" if pain_tier in ("severe", "very_severe") else "6–8 weeks"
            recs.append(f"Reassess in {interval}; track pain, walking ability, and treatment response.")
        else:
            recs.append("Close follow-up every 4–6 weeks; monitor functional trajectory and surgical readiness.")

        if conf_tier in ("very_low", "borderline") or margin < 10.0:
            recs.append("Repeat imaging recommended before escalating to invasive treatments.")

        if patient.age >= 75:
            recs.append("Review polypharmacy and falls risk at each visit.")
        if patient.diabetes or patient.hypertension:
            recs.append("Include comorbidity review (BP, HbA1c) at follow-up appointments.")

        return recs

    def _surgical_recs(self, patient, kl_grade, conf_tier, surgical_raw, risk_scores):
        recs = []

        if kl_grade < 3:
            recs.append("Surgery is not indicated at this radiographic stage — focus on conservative management.")
            return recs

        if conf_tier in ("very_low", "borderline"):
            recs.append(
                "Do not rush surgical referral at low/borderline CNN confidence — optimise conservative care and reimage."
            )
            return recs

        if kl_grade == 3:
            if patient.age < 50:
                recs.append(
                    "Consider earlier orthopaedic referral if functional goals are unmet after 3–6 months of conservative care."
                )
            elif patient.age < 65:
                recs.append(
                    "Orthopaedic referral if conservative management fails after 3–6 months; use shared decision-making."
                )
            elif patient.age < 75:
                recs.append("Favour continued conservative management unless function is severely impaired.")
            else:
                recs.append(
                    "Surgical candidacy requires careful evaluation of comorbidities, anaesthetic risk, and patient goals."
                )
        elif kl_grade == 4:
            if conf_tier == "high" and surgical_raw >= 60:
                recs.append(
                    "Strongly consider orthopaedic referral for arthroplasty if conservative measures are exhausted."
                )
                recs.append("Begin prehabilitation (quadriceps strengthening, weight optimisation) while awaiting review.")
            else:
                recs.append(
                    "Discuss partial or total knee replacement; balance surgical benefit against age and comorbidity profile."
                )

        if "High" in risk_scores.get("Surgical Need Probability", ""):
            recs.append("Risk model indicates elevated surgical probability — prioritise orthopaedic consultation.")

        return recs

    def _age_recs(self, patient, kl_grade, conf_tier, surgical_raw):
        recs = []

        if patient.age < 50:
            if kl_grade >= 3 and conf_tier in ("high", "moderate"):
                recs.append(
                    f"At age {patient.age}, consider earlier surgical candidacy if conservative goals are not met."
                )
            else:
                recs.append(
                    f"At age {patient.age}, focus on long-term joint health through strengthening and load management."
                )
        elif patient.age < 65:
            if kl_grade >= 3:
                recs.append(
                    f"At age {patient.age}, balance conservative and surgical options through shared decision-making."
                )
            else:
                recs.append(f"At age {patient.age}, structured exercise and lifestyle modification are the primary focus.")
        elif patient.age < 75:
            if kl_grade >= 3:
                recs.append(
                    f"At age {patient.age}, favour conservative management unless function is severely impaired."
                )
            else:
                recs.append(f"At age {patient.age}, encourage gentle exercise and address comorbidities affecting mobility.")
        else:
            if kl_grade >= 3 and conf_tier in ("high", "moderate"):
                recs.append(
                    f"At age {patient.age}, evaluate surgical candidacy against anaesthetic risk and functional goals."
                )
            recs.append(
                f"At age {patient.age}, prioritise fall-risk reduction, least-invasive options, and polypharmacy review."
            )

        return recs

    def _clinical_notes(self, patient, kl_grade, conf_tier, margin, classification):
        notes = []

        top3 = classification.get("top3", [])
        if len(top3) >= 2 and top3[1][1] > 25:
            notes.append(
                f"Alternative grade {top3[1][0]} has meaningful probability ({top3[1][1]:.1f}%) — keep differential in mind."
            )

        if kl_grade <= 1 and conf_tier in ("very_low", "borderline"):
            notes.append("Consider non-OA causes (patellofemoral pain, meniscal tear, early inflammatory arthritis).")

        if patient.gender == "Female" and patient.age >= 55 and kl_grade >= 2:
            notes.append("Post-menopausal status may accelerate cartilage loss — ensure adequate calcium/vitamin D intake.")

        if patient.activity_level == "High" and kl_grade >= 2:
            notes.append("High activity level with structural OA — balance fitness goals with joint preservation.")

        return notes


def _pdf_safe(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_next_steps(recommendations, risk_scores, urgency):
    next_steps = [
        "Consult your physician with this summary for comprehensive clinical review.",
    ]
    if recommendations.get("confidence_caveat"):
        next_steps.append(
            "Confirm radiographic grading clinically before committing to invasive treatment."
        )
    if urgency in ("High", "Urgent"):
        next_steps.append(
            "Schedule an expedited appointment within 1-2 weeks given symptom severity."
        )
    elif urgency == "Moderate":
        next_steps.append(
            "Schedule follow-up within 4-8 weeks to track functional scores (KOOS/WOMAC)."
        )
    else:
        next_steps.append("Routine follow-up in 8-12 weeks unless symptoms worsen.")
    if "High" in risk_scores.get("Surgical Need Probability", ""):
        next_steps.append("Prioritise an orthopaedic consultation regarding surgical options.")
    return next_steps


PDF_IMAGE_SIZE = 62


def _bgr_image_to_bytes(image_bgr: np.ndarray, max_size: int = 512, square: bool = True) -> BytesIO:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    if square:
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        rgb = rgb[y0:y0 + side, x0:x0 + side]

    side = rgb.shape[0]
    if side > max_size:
        rgb = cv2.resize(rgb, (max_size, max_size), interpolation=cv2.INTER_AREA)

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
    buf.seek(0)
    return buf


PDF_PRIMARY = (26, 54, 93)
PDF_PRIMARY_LIGHT = (232, 240, 248)
PDF_ACCENT = (0, 102, 153)
PDF_BORDER = (190, 200, 210)
PDF_TEXT_MUTED = (90, 90, 90)
PDF_WARN_BG = (255, 248, 235)
PDF_WARN_BORDER = (220, 160, 60)
PDF_URGENT = (180, 40, 40)
PDF_OK = (34, 120, 70)


class PatientReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_date = datetime.now().strftime("%d %b %Y, %H:%M")
        self.set_margins(14, 14, 14)
        self.set_auto_page_break(auto=True, margin=18)

    def content_width(self) -> float:
        return getattr(self, "epw", self.w - self.l_margin - self.r_margin)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_draw_color(*PDF_PRIMARY)
        self.set_line_width(0.4)
        self.line(self.l_margin, 12, self.w - self.r_margin, 12)
        self.set_xy(self.l_margin, 14)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*PDF_PRIMARY)
        self.cell(self.content_width() / 2, 5, _pdf_safe("KOA Clinical Decision Support Report"))
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*PDF_TEXT_MUTED)
        self.cell(self.content_width() / 2, 5, _pdf_safe(self.report_date), align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*PDF_BORDER)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*PDF_TEXT_MUTED)
        self.cell(
            self.content_width() / 2,
            4,
            _pdf_safe("AI-assisted report - not a substitute for clinical diagnosis."),
        )
        self.cell(self.content_width() / 2, 4, _pdf_safe(f"Page {self.page_no()}"), align="R")

    def _ensure_space(self, height: float):
        if self.get_y() + height > self.h - self.b_margin:
            self.add_page()

    def draw_cover_header(self, urgency: str):
        self.set_fill_color(*PDF_PRIMARY)
        self.rect(0, 0, 210, 38, style="F")
        self.set_xy(self.l_margin, 10)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, _pdf_safe("Knee Osteoarthritis Clinical Report"), ln=True)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(210, 225, 240)
        self.cell(
            0,
            6,
            _pdf_safe(f"AI-Assisted KL Grading & Clinical Decision Support  |  {self.report_date}"),
            ln=True,
        )
        self.set_y(42)
        self._draw_urgency_badge(urgency)

    def _draw_urgency_badge(self, urgency: str):
        colors = {
            "Low": PDF_OK,
            "Moderate": (200, 140, 20),
            "High": (210, 100, 30),
            "Urgent": PDF_URGENT,
        }
        self.set_fill_color(*colors.get(urgency, PDF_ACCENT))
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(255, 255, 255)
        badge_w = 32
        self.cell(badge_w, 7, _pdf_safe(f" {urgency} Priority "), fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def section_banner(self, title: str):
        self._ensure_space(14)
        self.set_x(self.l_margin)
        self.set_fill_color(*PDF_PRIMARY)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(255, 255, 255)
        self.cell(self.content_width(), 8, _pdf_safe(f"  {title}"), ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def subsection_title(self, title: str, x=None, w=None):
        self._ensure_space(10)
        x = self.l_margin if x is None else x
        w = self.content_width() if w is None else w
        self.set_xy(x, self.get_y())
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*PDF_PRIMARY)
        self.cell(w, 6, _pdf_safe(title), ln=True)
        self.set_text_color(0, 0, 0)

    def info_box(self, rows, x=None, w=None, fixed_height=None):
        x = x if x is not None else self.l_margin
        w = w if w is not None else self.content_width()
        row_h = 6.5
        box_h = fixed_height if fixed_height else len(rows) * row_h + 6
        y = self.get_y()
        self.set_fill_color(*PDF_PRIMARY_LIGHT)
        self.set_draw_color(*PDF_BORDER)
        self.rect(x, y, w, box_h, style="FD")
        self.set_xy(x + 3, y + 3)
        for label, value in rows:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*PDF_PRIMARY)
            self.cell(38, row_h, _pdf_safe(f"{label}"))
            self.set_font("Helvetica", "", 9)
            self.set_text_color(30, 30, 30)
            self.multi_cell(w - 44, row_h, _pdf_safe(str(value)))
            self.set_x(x + 3)
        self.set_xy(x, y + box_h)

    def two_column_boxes(self, left_title, left_rows, right_title, right_rows):
        gap = 4
        col_w = (self.content_width() - gap) / 2
        title_h = 7
        left_h = len(left_rows) * 6.5 + 6
        right_h = len(right_rows) * 6.5 + 6
        box_h = max(left_h, right_h)
        self._ensure_space(title_h + box_h + 6)

        y0 = self.get_y()
        y_box = y0 + title_h

        self.set_xy(self.l_margin, y0)
        self.subsection_title(left_title, x=self.l_margin, w=col_w)
        self.set_xy(self.l_margin, y_box)
        self.info_box(left_rows, x=self.l_margin, w=col_w, fixed_height=box_h)

        self.set_xy(self.l_margin + col_w + gap, y0)
        self.subsection_title(right_title, x=self.l_margin + col_w + gap, w=col_w)
        self.set_xy(self.l_margin + col_w + gap, y_box)
        self.info_box(right_rows, x=self.l_margin + col_w + gap, w=col_w, fixed_height=box_h)

        self.set_y(y_box + box_h + 4)

    def highlight_box(self, text: str, style: str = "info"):
        self._ensure_space(20)
        if style == "warning":
            bg, border = PDF_WARN_BG, PDF_WARN_BORDER
        else:
            bg, border = PDF_PRIMARY_LIGHT, PDF_ACCENT
        y = self.get_y()
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 9.5)
        lines = self.multi_cell(self.content_width() - 8, 5, _pdf_safe(text), split_only=True)
        box_h = max(len(lines) * 5 + 8, 14)
        self.set_fill_color(*bg)
        self.set_draw_color(*border)
        self.rect(self.l_margin, y, self.content_width(), box_h, style="FD")
        self.set_xy(self.l_margin + 4, y + 4)
        self.multi_cell(self.content_width() - 8, 5, _pdf_safe(text))
        self.set_y(y + box_h + 4)

    def image_row(self, panels, img_size=PDF_IMAGE_SIZE):
        if not panels:
            return
        n = len(panels)
        gap = 6
        caption_h = 8
        square = img_size
        panel_w = square + 6
        panel_h = square + caption_h + 8
        row_w = n * panel_w + (n - 1) * gap
        start_x = self.l_margin + max((self.content_width() - row_w) / 2, 0)
        total_h = panel_h + 2
        self._ensure_space(total_h)

        y = self.get_y()

        for i, (img_bgr, caption) in enumerate(panels):
            x = start_x + i * (panel_w + gap)
            x_img = x + (panel_w - square) / 2
            self.set_draw_color(*PDF_BORDER)
            self.set_fill_color(255, 255, 255)
            self.rect(x, y, panel_w, panel_h, style="FD")
            try:
                self.image(
                    _bgr_image_to_bytes(img_bgr, square=True),
                    x=x_img,
                    y=y + 3,
                    w=square,
                    h=square,
                    keep_aspect_ratio=False,
                )
            except Exception:
                self.set_xy(x_img, y + square / 2)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*PDF_TEXT_MUTED)
                self.cell(square, 5, _pdf_safe("Image unavailable"), align="C")
            self.set_xy(x, y + square + 6)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*PDF_PRIMARY)
            self.cell(panel_w, caption_h, _pdf_safe(caption), align="C")

        self.set_y(y + total_h)

    def grade_result_card(self, grade_label: str, confidence: float, model_name: str):
        self._ensure_space(28)
        y = self.get_y()
        w = self.content_width()
        self.set_fill_color(*PDF_PRIMARY)
        self.rect(self.l_margin, y, w, 26, style="F")
        self.set_xy(self.l_margin + 5, y + 4)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(210, 225, 240)
        self.cell(60, 5, _pdf_safe("PREDICTED KL GRADE"))
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_xy(self.l_margin + 5, y + 10)
        self.cell(w - 80, 10, _pdf_safe(grade_label))
        self.set_xy(self.l_margin + w - 70, y + 5)
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(255, 255, 255)
        self.cell(30, 12, _pdf_safe(f"{confidence:.1f}%"), align="R")
        self.set_xy(self.l_margin + w - 70, y + 17)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(210, 225, 240)
        self.cell(65, 5, _pdf_safe(f"Model: {model_name}"), align="R")
        self.set_y(y + 30)

    def risk_score_grid(self, scores: dict):
        self._ensure_space(44)
        items = [
            ("Progression Risk", scores["Progression Risk Score"]),
            ("Functional Disability", scores["Functional Disability Risk"]),
            ("Obesity Risk", scores["Obesity Risk"]),
            ("Surgical Need", scores["Surgical Need Probability"]),
        ]
        raw = scores.get("_raw", {})
        raw_keys = ["progression", "disability", "obesity", "surgical"]

        gap = 4
        col_w = (self.content_width() - gap) / 2
        row_h = 20
        y0 = self.get_y()

        for idx, (label, value) in enumerate(items):
            col = idx % 2
            row = idx // 2
            x = self.l_margin + col * (col_w + gap)
            y = y0 + row * (row_h + gap)

            if "High" in value:
                fill = (255, 235, 235)
                accent = PDF_URGENT
            elif "Moderate" in value:
                fill = (255, 245, 225)
                accent = (200, 130, 20)
            else:
                fill = (235, 245, 235)
                accent = PDF_OK

            self.set_fill_color(*fill)
            self.set_draw_color(*accent)
            self.rect(x, y, col_w, row_h, style="FD")

            score_text = value
            if idx < len(raw_keys) and raw_keys[idx] in raw:
                numeric = raw[raw_keys[idx]]
                level = value.split("(")[0].strip() if "(" in value else value
                score_text = f"{level}  {numeric:.1f}/100"

            self.set_xy(x + 3, y + 3)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*PDF_PRIMARY)
            self.cell(col_w - 6, 5, _pdf_safe(label))

            self.set_xy(x + 3, y + 10)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*accent)
            self.cell(col_w - 6, 8, _pdf_safe(score_text), align="R")

        self.set_y(y0 + 2 * (row_h + gap) + 2)

    def probability_table(self, all_probs, class_labels: dict):
        self._ensure_space(40)
        self.subsection_title("Full Classification Probabilities")
        col_w = self.content_width()
        row_h = 7
        headers = ["KL Grade", "Probability", "Bar"]
        widths = [col_w * 0.45, col_w * 0.18, col_w * 0.37]

        self.set_fill_color(*PDF_PRIMARY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        for w, h in zip(widths, headers):
            self.cell(w, row_h, _pdf_safe(h), border=1, fill=True)
        self.ln()

        for idx, prob in enumerate(all_probs):
            pct = prob * 100.0
            fill = PDF_PRIMARY_LIGHT if idx % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*fill)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", 9)
            grade_label = class_labels.get(idx, f"Class {idx}")
            self.cell(widths[0], row_h, _pdf_safe(grade_label), border=1, fill=True)
            self.cell(widths[1], row_h, _pdf_safe(f"{pct:.2f}%"), border=1, fill=True)
            bar_x = self.get_x() + 2
            bar_y = self.get_y() + 2
            self.cell(widths[2], row_h, "", border=1, fill=True)
            bar_max_w = widths[2] - 4
            bar_w = max(bar_max_w * (pct / 100.0), 0.5 if pct > 0 else 0)
            self.set_fill_color(*PDF_ACCENT)
            self.rect(bar_x, bar_y, bar_w, row_h - 4, style="F")
            self.ln()
        self.ln(3)

    def bullet_section(self, title: str, items: list):
        if not items:
            return
        self._ensure_space(12 + len(items) * 5)
        self.subsection_title(title)
        self.set_x(self.l_margin + 2)
        self.set_font("Helvetica", "", 9)
        for item in items:
            self.set_x(self.l_margin + 2)
            self.set_text_color(*PDF_ACCENT)
            self.cell(4, 5, _pdf_safe("-"))
            self.set_text_color(40, 40, 40)
            self.multi_cell(self.content_width() - 6, 5, _pdf_safe(item))
        self.ln(2)

    def numbered_section(self, title: str, items: list):
        if not items:
            return
        self._ensure_space(12 + len(items) * 5)
        self.subsection_title(title)
        self.set_font("Helvetica", "", 9)
        for i, item in enumerate(items, 1):
            self.set_x(self.l_margin + 2)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*PDF_PRIMARY)
            self.cell(6, 5, _pdf_safe(f"{i}."))
            self.set_font("Helvetica", "", 9)
            self.set_text_color(40, 40, 40)
            self.multi_cell(self.content_width() - 8, 5, _pdf_safe(item))
        self.ln(2)


def generate_patient_report_pdf(
    patient: PatientProfile,
    result: dict,
    risk_scores: dict,
    recommendations: dict,
    next_steps: list,
    model_name: str,
    class_labels: dict,
    original_image_bgr: np.ndarray | None = None,
    processed_image_bgr: np.ndarray | None = None,
    gradcam_heatmap_bgr: np.ndarray | None = None,
    gradcam_overlay_bgr: np.ndarray | None = None,
) -> bytes:
    if not FPDF_AVAILABLE:
        raise ImportError("PDF export requires fpdf2. Install with: pip install fpdf2")

    pdf = PatientReportPDF()
    pdf.add_page()

    comorbidities = []
    if patient.prev_injury:
        comorbidities.append("Previous knee injury")
    if patient.diabetes:
        comorbidities.append("Diabetes")
    if patient.hypertension:
        comorbidities.append("Hypertension")
    comorbidity_text = ", ".join(comorbidities) if comorbidities else "None reported"

    pdf.draw_cover_header(recommendations["urgency"])
    pdf.grade_result_card(result["predicted_label"], result["confidence"], model_name)

    pdf.section_banner("Patient Demographics & Symptoms")
    pdf.two_column_boxes(
        "Clinical Profile",
        [
            ("Age", f"{patient.age} years"),
            ("Gender", patient.gender),
            ("Height", f"{patient.height_cm} cm"),
            ("Weight", f"{patient.weight_kg} kg"),
            ("BMI", f"{patient.bmi:.1f} kg/m2"),
            ("Activity", patient.activity_level),
        ],
        "Symptoms & History",
        [
            ("Pain (VAS)", f"{patient.pain_score} / 10"),
            ("Walking Difficulty", f"{patient.walking_diff_score} / 10"),
            ("Comorbidities", comorbidity_text),
            ("Confidence Tier", recommendations["confidence_tier"].replace("_", " ").title()),
            ("Care Urgency", recommendations["urgency"]),
        ],
    )

    xray_panels = []
    if original_image_bgr is not None:
        xray_panels.append((original_image_bgr, "Original X-ray"))
    if processed_image_bgr is not None:
        xray_panels.append((processed_image_bgr, "Preprocessed X-ray"))

    if xray_panels:
        pdf.section_banner("Radiological Imaging")
        pdf.image_row(xray_panels, img_size=PDF_IMAGE_SIZE)

    gradcam_panels = []
    if gradcam_heatmap_bgr is not None:
        gradcam_panels.append((gradcam_heatmap_bgr, "Grad-CAM Heatmap (Red = High Focus)"))
    if gradcam_overlay_bgr is not None:
        gradcam_panels.append((gradcam_overlay_bgr, "Grad-CAM Overlay on X-ray"))

    if gradcam_panels:
        pdf.section_banner("AI Explainability - Grad-CAM")
        pdf.image_row(gradcam_panels, img_size=PDF_IMAGE_SIZE)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*PDF_TEXT_MUTED)
        pdf.multi_cell(
            pdf.content_width(),
            4,
            _pdf_safe(
                "Grad-CAM visualises where the AI model focused when making its prediction. "
                "Red areas = highest influence; blue = lowest."
            ),
        )
        pdf.ln(3)

    pdf.section_banner("Clinical Assessment")
    pdf.highlight_box(recommendations["clinical_summary"], style="info")
    if recommendations.get("confidence_caveat"):
        pdf.highlight_box(recommendations["confidence_caveat"], style="warning")

    pdf.section_banner("Risk Assessment")
    pdf.risk_score_grid(risk_scores)

    all_probs = result.get("all_probabilities", [])
    if all_probs:
        pdf.probability_table(all_probs, class_labels)

    pdf.section_banner("Priority Actions")
    pdf.numbered_section("", recommendations.get("priority_actions", []))

    rec_sections = [
        ("Pain Management", "Pain Management Recommendations"),
        ("Exercise & Physiotherapy", "Exercise Recommendations"),
        ("Weight Management", "Weight Management Recommendations"),
        ("Lifestyle Modifications", "Lifestyle Recommendations"),
        ("Medical Follow-Up", "Medical Follow-Up Recommendations"),
        ("Surgical Consultation", "Surgical Consultation Recommendations"),
        ("Age-Specific Guidance", "Age-Specific Guidance"),
    ]

    pdf.section_banner("Personalized Treatment Recommendations")
    for title, key in rec_sections:
        items = recommendations.get(key, [])
        pdf.bullet_section(title, items)

    if recommendations.get("Clinical Notes"):
        pdf.section_banner("Additional Clinical Notes")
        pdf.bullet_section("", recommendations["Clinical Notes"])

    pdf.section_banner("Recommended Next Steps")
    pdf.bullet_section("", next_steps)

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*PDF_TEXT_MUTED)
    pdf.multi_cell(
        pdf.content_width(),
        4,
        _pdf_safe(
            "DISCLAIMER: This document was generated by an AI-assisted knee osteoarthritis grading system. "
            "It is intended to support - not replace - clinical judgment. All findings must be validated "
            "by a qualified healthcare professional before treatment decisions are made."
        ),
    )

    return bytes(pdf.output())


def run_full_analysis(experiment_id, model_id, gamma, use_gradcam, image_bytes):
    file_arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    original_image = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError("Failed to decode the uploaded image.")

    processed_image = process_single_image(original_image, gamma=gamma)

    model, model_display_name, _profile = load_cached_model(experiment_id, model_id)
    class_labels = get_class_labels(EXPERIMENT_PROFILES[experiment_id]["num_classes"])
    result = predict(model, processed_image, IMG_SIZE, class_labels)

    gradcam_overlay = None
    gradcam_heatmap = None
    if use_gradcam and model_supports_gradcam(model_id):
        tensor_img = prepare_image(processed_image, IMG_SIZE).to(DEVICE)
        heatmap = generate_gradcam_heatmap(
            model, model_id, tensor_img, result["predicted_class"], out_size=IMG_SIZE,
        )
        if heatmap is not None:
            gradcam_heatmap = get_gradcam_heatmap_image(processed_image, heatmap)
            gradcam_overlay = get_gradcam_overlay(processed_image, heatmap)

    return {
        "original_image": original_image,
        "processed_image": processed_image,
        "result": result,
        "model_display_name": model_display_name,
        "gradcam_overlay": gradcam_overlay,
        "gradcam_heatmap": gradcam_heatmap,
    }


def get_cached_analysis(experiment_id, model_id, gamma, use_gradcam, image_bytes):
    key = (
        experiment_id,
        model_id,
        round(float(gamma), 2),
        bool(use_gradcam),
        hashlib.md5(image_bytes).hexdigest(),
    )
    if st.session_state.get("analysis_key") == key and "analysis" in st.session_state:
        return st.session_state["analysis"]

    analysis = run_full_analysis(experiment_id, model_id, gamma, use_gradcam, image_bytes)
    st.session_state["analysis_key"] = key
    st.session_state["analysis"] = analysis
    return analysis


def main():
    st.set_page_config(page_title="KOA Severity Grading", page_icon="🦴", layout="centered")

    st.title("🦴 KOA Severity Grading")
    st.markdown(
        "Upload a knee X-ray for **Kellgren–Lawrence (KL)** grading using GAN-augmented deep learning "
        "(project steps 7–10). Includes preprocessing, classification, Grad-CAM, and clinical decision support."
    )

    exp_options = cached_experiment_options()
    exp_label_by_id = {opt.experiment_id: opt.label for opt in exp_options}
    exp_labels = [opt.label for opt in exp_options]
    default_exp_label = exp_label_by_id[DEFAULT_EXPERIMENT_ID]


    with st.sidebar:
        st.header("⚙️ Model pipeline")
        selected_exp_label = st.selectbox(
            "Training experiment",
            exp_labels,
            index=exp_labels.index(default_exp_label),
            help="Loads checkpoints from the matching experiment folder (5 Classes, 4 Classes, etc.).",
        )
        experiment_id = {opt.label: opt.experiment_id for opt in exp_options}[selected_exp_label]
        profile = EXPERIMENT_PROFILES[experiment_id]
        exp_meta = next(o for o in exp_options if o.experiment_id == experiment_id)

        st.caption(exp_meta.subtitle)
        best_name, best_acc = cached_best_ensemble(experiment_id)
        if best_name is not None and best_acc is not None:
            st.caption(f"**Best reported:** {best_name} {best_acc:.2f}%")
        else:
            st.caption(f"**Best reported:** {profile['best_method']}")

        model_options = cached_model_options(experiment_id)
        if not model_options:
            st.error(f"No checkpoints found in `{profile['folder']}`.")
            return

        model_label_by_id = {opt.model_id: opt.label for opt in model_options}
        model_labels = [opt.label for opt in model_options]
        default_model_id = profile.get("default_model_id")
        if default_model_id not in model_label_by_id:
            default_model_id = model_options[0].model_id
        default_model_label = model_label_by_id[default_model_id]

        selected_model_label = st.selectbox(
            "Classifier",
            model_labels,
            index=model_labels.index(default_model_label),
            help="⭐ = recommended ensemble or fusion. Other options are individual CNN/ViT backbones.",
        )
        selected_model_id = {opt.label: opt.model_id for opt in model_options}[selected_model_label]
        selected_model_meta = next(o for o in model_options if o.model_id == selected_model_id)
        st.caption(selected_model_meta.subtitle)

        class_labels = get_class_labels(profile["num_classes"])
        class_mode = "4-class (grades 0+1 merged)" if profile["num_classes"] == 4 else "5-class (KL 0–4)"
        st.markdown(f"**Output:** {class_mode}")

        st.markdown("---")
        st.header("🖼️ Preprocessing")
        gamma_value = st.slider("Gamma correction (γ)", min_value=0.1, max_value=3.0, value=1.2, step=0.1)

        st.markdown("---")
        st.header("🔬 Analysis")
        use_gradcam = st.checkbox(
            "Grad-CAM heatmap",
            value=True,
            disabled=not model_supports_gradcam(selected_model_id),
            help="Shows where the model focused. Fusion and ensemble use averaged backbone heatmaps.",
        )

        st.markdown("---")
        st.markdown(f"**Device:** `{DEVICE}`")


    st.header("📋 Patient Clinical Information")
    with st.expander("Enter Patient Details", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=60)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=75)
            pain = st.slider("Pain Score (0-10)", 0, 10, 5)
            walking_diff = st.slider("Walking Difficulty Score (0-10)", 0, 10, 3)
        with col3:
            st.markdown("**Comorbidities & History**")
            prev_injury = st.checkbox("Previous Knee Injury")
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
            activity_level = st.selectbox("Activity Level", ["Low", "Moderate", "High"])

        patient = PatientProfile(age, gender, height, weight, pain, walking_diff, prev_injury, diabetes, hypertension,
                                 activity_level)
        st.info(f"**Calculated BMI:** {patient.bmi:.1f} kg/m²")

    st.markdown("---")


    uploaded_file = st.file_uploader("📂 Upload X-ray Image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        try:
            with st.spinner("Analyzing X-ray..."):
                analysis = get_cached_analysis(
                    experiment_id, selected_model_id, gamma_value, use_gradcam, image_bytes
                )

            original_image = analysis["original_image"]
            processed_image = analysis["processed_image"]
            result = analysis["result"]
            model_display_name = analysis["model_display_name"]
            gradcam_overlay = analysis["gradcam_overlay"]
            gradcam_heatmap = analysis["gradcam_heatmap"]


            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Original Image")
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), width=350)

            with col2:
                st.markdown("#### Processed Image")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), width=350)

            st.markdown("---")


            st.subheader("🔬 Classification Result")
            st.caption(f"**Model:** {selected_model_meta.label} · **Pipeline:** {exp_meta.label}")


            st.success(
                f"**Predicted Grade:** {result['predicted_label']}  \n**Confidence:** {result['confidence']:.2f}%")


            if use_gradcam and model_supports_gradcam(selected_model_id):
                gradcam_caption = "Red = High Focus, Blue = Low Focus"
                if selected_model_id == FUSION_MODEL_ID:
                    gradcam_caption += " · averaged across fusion backbones"
                elif selected_model_id == ENSEMBLE_MODEL_ID:
                    gradcam_caption += " · accuracy-weighted across ensemble backbones"
                if gradcam_overlay is not None:
                    st.markdown("#### Grad-CAM Focus Heatmap")
                    st.image(
                        cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB),
                        width=350,
                        caption=gradcam_caption,
                    )
                else:
                    st.warning("Grad-CAM could not be generated for this model.")
            st.markdown("---")


            st.write("### Top-3 Predictions")
            for label, prob in result["top3"]:
                col_label, col_prob = st.columns([1, 4])
                col_label.write(f"**{label}**")
                col_prob.progress(prob / 100.0, text=f"{prob:.2f}%")

            with st.expander("View All Probabilities"):
                for idx, prob in enumerate(result["all_probabilities"]):
                    st.write(f"- {class_labels[idx]}: **{prob * 100:.2f}%**")


            st.markdown("---")
            st.subheader("🏥 Hybrid Clinical Decision Support System (CDSS)")

            kl_grade = result['predicted_class']
            risk_engine = RiskAssessmentEngine()
            recomm_engine = RecommendationEngine()

            risk_scores = risk_engine.compute_scores(patient, kl_grade)
            recommendations = recomm_engine.generate(
                patient, kl_grade, risk_scores, classification=result
            )

            urgency = recommendations["urgency"]
            urgency_colors = {
                "Low": "🟢",
                "Moderate": "🟡",
                "High": "🟠",
                "Urgent": "🔴",
            }

            st.markdown("#### Clinical Assessment")
            st.info(recommendations["clinical_summary"])

            col_u, col_c = st.columns(2)
            with col_u:
                st.metric("Care Urgency", f"{urgency_colors.get(urgency, '⚪')} {urgency}")
            with col_c:
                st.metric("Model Confidence Tier", recommendations["confidence_tier"].replace("_", " ").title())

            if recommendations["confidence_caveat"]:
                st.warning(recommendations["confidence_caveat"])

            st.markdown("#### Priority Actions")
            for i, action in enumerate(recommendations["priority_actions"], 1):
                st.markdown(f"{i}. **{action}**")

            st.markdown("#### Patient Summary")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Age", f"{patient.age} yrs")
            summary_cols[1].metric("BMI", f"{patient.bmi:.1f}")
            summary_cols[2].metric("Pain Score", f"{patient.pain_score}/10")
            summary_cols[3].metric("Walking Difficulty", f"{patient.walking_diff_score}/10")
            st.caption(
                f"KL Grade **{kl_grade}** ({result['predicted_label']}) · "
                f"CNN confidence **{result['confidence']:.1f}%** · "
                f"Activity level **{patient.activity_level}**"
            )

            st.markdown("#### Risk Assessment")
            risk_cols = st.columns(2)
            risk_cols[0].write(f"- **Progression Risk:** {risk_scores['Progression Risk Score']}")
            risk_cols[0].write(f"- **Functional Disability Risk:** {risk_scores['Functional Disability Risk']}")
            risk_cols[1].write(f"- **Obesity Risk:** {risk_scores['Obesity Risk']}")
            risk_cols[1].write(f"- **Surgical Need Probability:** {risk_scores['Surgical Need Probability']}")

            st.markdown("#### Personalized Recommendations")

            rec_sections = [
                ("💊 Pain Management", "Pain Management Recommendations"),
                ("🏃 Exercise", "Exercise Recommendations"),
                ("🥗 Weight Management", "Weight Management Recommendations"),
                ("🌿 Lifestyle", "Lifestyle Recommendations"),
                ("📅 Medical Follow-Up", "Medical Follow-Up Recommendations"),
                ("🏥 Surgical Consultation", "Surgical Consultation Recommendations"),
                ("👤 Age-Specific Guidance", "Age-Specific Guidance"),
            ]

            for title, key in rec_sections:
                items = recommendations.get(key, [])
                if items:
                    with st.expander(title, expanded=key in (
                        "Pain Management Recommendations",
                        "Exercise Recommendations",
                        "Medical Follow-Up Recommendations",
                    )):
                        for r in items:
                            st.write(f"- {r}")

            if recommendations["Clinical Notes"]:
                st.markdown("#### Clinical Notes")
                for note in recommendations["Clinical Notes"]:
                    st.write(f"- {note}")

            st.markdown("#### Next Steps")
            next_steps = build_next_steps(recommendations, risk_scores, urgency)
            for step in next_steps:
                st.write(f"- {step}")

            st.markdown("---")
            st.subheader("📄 Patient Report")

            if not FPDF_AVAILABLE:
                st.info("Install `fpdf2` to enable PDF export: `pip install fpdf2`")
            else:
                if st.button("Generate Patient Report (PDF)", type="primary", key="generate_pdf_btn"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            pdf_bytes = generate_patient_report_pdf(
                                patient=patient,
                                result=result,
                                risk_scores=risk_scores,
                                recommendations=recommendations,
                                next_steps=next_steps,
                                model_name=f"{model_display_name} · {exp_meta.label}",
                                class_labels=class_labels,
                                original_image_bgr=original_image,
                                processed_image_bgr=processed_image,
                                gradcam_heatmap_bgr=gradcam_heatmap,
                                gradcam_overlay_bgr=gradcam_overlay,
                            )
                            st.session_state["patient_report_pdf"] = pdf_bytes
                            st.session_state["patient_report_filename"] = (
                                f"KOA_Patient_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            )
                            st.success("PDF report ready — click Download below.")
                        except Exception as pdf_err:
                            st.error(f"Could not generate PDF report: {pdf_err}")

                if st.session_state.get("patient_report_pdf"):
                    st.download_button(
                        label="Download Patient Report (PDF)",
                        data=st.session_state["patient_report_pdf"],
                        file_name=st.session_state.get(
                            "patient_report_filename", "KOA_Patient_Report.pdf"
                        ),
                        mime="application/pdf",
                        key="download_pdf_btn",
                    )

        except FileNotFoundError as e:
            st.error(f"Model file missing: {e}")
        except ImportError as e:
            st.error(f"Missing dependency: {e}")
            st.info("Install required packages: `pip install torch torchvision timm streamlit opencv-python`")
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            st.error(f"Classification failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            with st.expander("Technical details"):
                st.exception(e)


if __name__ == "__main__":
    main()
