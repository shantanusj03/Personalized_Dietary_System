from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, time
@dataclass
class UserProfile:
    user_id: str
    age: int
    sex: Literal["male", "female"]
    height_cm: float
    weight_kg: float
    activity_level: Literal["low", "moderate", "high"] = "moderate"
    sleep_hours: Optional[float] = None
    conditions: List[str] = field(default_factory=list)
    vegetarian: Optional[bool] = None
    allergies: List[str] = field(default_factory=list)
    @property
    def bmi(self) -> float:
        h_m = self.height_cm / 100.0
        return round(self.weight_kg / (h_m * h_m), 1)
@dataclass
class FoodItem:
    name: str
    grams: float
    calories: float
    carbs_g: float
    protein_g: float
    fat_g: float
    sugar_g: float
    fiber_g: float
    sodium_mg: float
    sat_fat_g: Optional[float] = None
    trans_fat_g: Optional[float] = None
    cholesterol_mg: Optional[float] = None
    tags: List[str] = field(default_factory=list)
@dataclass
class DailyTotals:
    calories: float = 0.0
    carbs_g: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    sugar_g: float = 0.0
    fiber_g: float = 0.0
    sodium_mg: float = 0.0
    sat_fat_g: float = 0.0
    trans_fat_g: float = 0.0
    cholesterol_mg: float = 0.0
@dataclass
class PersonalLimits:
    calorie_target: float
    protein_target_g: float
    sugar_limit_g: float
    sodium_limit_mg: float
    fiber_target_g: float
    fat_cap_g: float
    sat_fat_cap_g: float
    trans_fat_cap_g: float
    cholesterol_cap_mg: float
@dataclass
class Evaluation:
    category: Literal["Acceptable", "Warning", "Not Recommended"]
    reasons: List[str]
    numbers: Dict[str, Any]
    alternatives: List[str]
    tips: List[str]
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
ACTIVITY_FACTORS = {
    "low": 1.2,
    "moderate": 1.45,
    "high": 1.7,
}
ALTERNATIVES_MAP: Dict[str, List[str]] = {
    "Gulab Jamun": ["Fruit Salad (no added sugar)", "Sugar-free Rasgulla", "Baked Sandesh", "Unsweetened Yogurt with Nuts"],
    "Jalebi": ["Roasted Nuts (unsalted)", "Fruit Chaat (no syrup)", "Dark Chocolate (small piece)"],
    "Kheer": ["Dalia/Poha (less sugar)", "Kheer with stevia (small)", "Plain Phirni (small, low sugar)"],
    "Samosa": ["Baked Samosa", "Steamed Dhokla", "Roasted Chana", "Sprouts Chaat"],
    "Pakora": ["Grilled Paneer Tikka", "Steamed Corn", "Bhel with sprouts (less sev)"],
    "Pani Puri": ["Sprout Bhel (less salt)", "Dahi Puri (curd-based, less chutney)", "Cucumber Slices with Hummus"],
    "White Rice": ["Brown Rice", "Quinoa (small portion)", "Mixed Millets"],
    "Butter Naan": ["Phulka/Roti (no ghee)", "Multigrain Roti"],
    "Chicken Biryani": ["Grilled Chicken + Brown Rice (small)", "Chicken Curry (less oil) + Salad"],
    "Paneer Butter Masala": ["Kadhai Paneer (less oil)", "Palak Paneer (less cream)", "Chana Masala"],
    "Sweet Lassi": ["Chaas/Buttermilk (unsalted)", "Plain Lassi (no sugar)", "Lemon Water"],
    "Thick Shake": ["Milk + Unsweetened Cocoa (small)", "Cold Coffee (no sugar)"],
}
TAG_ALTERNATIVES = {
    "sweet": ["Fruit Salad (no added sugar)", "Sugar-free Sandesh", "Unsweetened Yogurt + Seeds"],
    "fried": ["Grilled Paneer/Chicken", "Steamed Dhokla/Idli", "Roasted Chana"],
    "refined_carb": ["Brown Rice", "Millet/Quinoa", "Whole-wheat Roti"],
    "beverage": ["Buttermilk (unsalted)", "Lemon Water", "Plain Lassi (no sugar)"],
    "salty": ["Fresh Salad", "Steamed Veggies + Lemon", "Sprouts Chaat (less salt)"]
}
BAND_WARNING = 0.8
BAND_NOTREC = 1.0
NIGHT_HEAVY_AFTER = time(21, 0)
def mifflin_st_jeor_bmr(sex: str, weight_kg: float, height_cm: float, age: int) -> float:
    if sex == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
def tdee_from_activity(bmr: float, activity_level: str) -> float:
    return bmr * ACTIVITY_FACTORS.get(activity_level, ACTIVITY_FACTORS["moderate"])
def derive_personal_limits(profile: UserProfile) -> PersonalLimits:
    bmr = mifflin_st_jeor_bmr(profile.sex, profile.weight_kg, profile.height_cm, profile.age)
    tdee = tdee_from_activity(bmr, profile.activity_level)
    if profile.bmi >= 30:
        calorie_target = tdee * 0.8
    elif profile.bmi >= 25:
        calorie_target = tdee * 0.9
    elif profile.bmi < 18.5:
        calorie_target = tdee * 1.1
    else:
        calorie_target = tdee
    protein_multiplier = 0.8
    if profile.activity_level == "high":
        protein_multiplier = 1.2
    elif profile.activity_level == "moderate":
        protein_multiplier = 1.0
    if profile.age >= 60:
        protein_multiplier = max(protein_multiplier, 1.1)
    protein_target_g = profile.weight_kg * protein_multiplier
    sugar_limit_g = 50.0
    sodium_limit_mg = 2000.0
    if "diabetes" in profile.conditions:
        sugar_limit_g = 25.0
    if "hypertension" in profile.conditions:
        sodium_limit_mg = 1500.0
    fiber_target_g = max(18.0, 14.0 * (calorie_target / 1000.0))
    fat_cap_g = (0.30 * calorie_target) / 9.0
    sat_fat_cap_g = (0.10 * calorie_target) / 9.0
    trans_fat_cap_g = 2.0
    cholesterol_cap_mg = 300.0
    if "dyslipidemia" in profile.conditions:
        sat_fat_cap_g = min(sat_fat_cap_g, 15.0)
        trans_fat_cap_g = 1.0
        cholesterol_cap_mg = 200.0
    return PersonalLimits(
        calorie_target=round(calorie_target),
        protein_target_g=round(protein_target_g, 1),
        sugar_limit_g=round(sugar_limit_g, 1),
        sodium_limit_mg=round(sodium_limit_mg),
        fiber_target_g=round(fiber_target_g, 1),
        fat_cap_g=round(fat_cap_g, 1),
        sat_fat_cap_g=round(sat_fat_cap_g, 1),
        trans_fat_cap_g=trans_fat_cap_g,
        cholesterol_cap_mg=cholesterol_cap_mg,
    )
def sum_daily_totals(log_items: List[FoodItem]) -> DailyTotals:
    tot = DailyTotals()
    for it in log_items:
        tot.calories += it.calories
        tot.carbs_g += it.carbs_g
        tot.protein_g += it.protein_g
        tot.fat_g += it.fat_g
        tot.sugar_g += it.sugar_g
        tot.fiber_g += it.fiber_g
        tot.sodium_mg += it.sodium_mg
        if it.sat_fat_g is not None:
            tot.sat_fat_g += it.sat_fat_g
        if it.trans_fat_g is not None:
            tot.trans_fat_g += it.trans_fat_g
        if it.cholesterol_mg is not None:
            tot.cholesterol_mg += it.cholesterol_mg
    for k, v in tot.__dict__.items():
        setattr(tot, k, round(v, 1))
    return tot
def _remaining(used: float, limit: float) -> float:
    return max(0.0, round(limit - used, 1))
def current_time_is_late(now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    return now.time() >= NIGHT_HEAVY_AFTER
def evaluate(
    food: FoodItem,
    profile: UserProfile,
    today_log: List[FoodItem],
    now: Optional[datetime] = None
) -> Evaluation:
    limits = derive_personal_limits(profile)
    totals = sum_daily_totals(today_log)
    numbers = {
        "bmi": profile.bmi,
        "calorie_target": limits.calorie_target,
        "protein_target_g": limits.protein_target_g,
        "sugar_limit_g": limits.sugar_limit_g,
        "sodium_limit_mg": limits.sodium_limit_mg,
        "fiber_target_g": limits.fiber_target_g,
        "fat_cap_g": limits.fat_cap_g,
        "sat_fat_cap_g": limits.sat_fat_cap_g,
        "trans_fat_cap_g": limits.trans_fat_cap_g,
        "cholesterol_cap_mg": limits.cholesterol_cap_mg,
        "today_so_far": {
            "calories": totals.calories,
            "sugar_g": totals.sugar_g,
            "sodium_mg": totals.sodium_mg,
            "protein_g": totals.protein_g,
            "fiber_g": totals.fiber_g,
            "fat_g": totals.fat_g,
            "sat_fat_g": totals.sat_fat_g,
        },
        "this_food": {
            "calories": food.calories,
            "sugar_g": food.sugar_g,
            "sodium_mg": food.sodium_mg,
            "protein_g": food.protein_g,
            "fiber_g": food.fiber_g,
            "fat_g": food.fat_g,
            "sat_fat_g": food.sat_fat_g or 0.0,
        }
    }
    remain = {
        "calories": _remaining(totals.calories, limits.calorie_target),
        "sugar_g": _remaining(totals.sugar_g, limits.sugar_limit_g),
        "sodium_mg": _remaining(totals.sodium_mg, limits.sodium_limit_mg),
        "fat_g": _remaining(totals.fat_g, limits.fat_cap_g),
        "sat_fat_g": _remaining(totals.sat_fat_g, limits.sat_fat_cap_g),
        "fiber_g": max(0.0, round(limits.fiber_target_g - totals.fiber_g, 1)),
        "protein_g": max(0.0, round(limits.protein_target_g - totals.protein_g, 1)),
    }
    def classify(metric: str, add_value: float, limit: float, remaining: float) -> Optional[str]:
        if limit <= 0:
            return None
        if remaining <= 0 and add_value > 0:
            return "Not Recommended"
        ratio = (limit - remaining + add_value) / limit
        if ratio > BAND_NOTREC:
            return "Not Recommended"
        elif ratio > BAND_WARNING:
            return "Warning"
        return None
    reasons: List[str] = []
    flags: List[str] = []
    sugar_flag = classify("sugar_g", food.sugar_g, limits.sugar_limit_g, remain["sugar_g"])
    sodium_flag = classify("sodium_mg", food.sodium_mg, limits.sodium_limit_mg, remain["sodium_mg"])
    calorie_flag = classify("calories", food.calories, limits.calorie_target, remain["calories"])
    fat_flag = classify("fat_g", food.fat_g, limits.fat_cap_g, remain["fat_g"])
    sat_fat_flag = classify("sat_fat_g", (food.sat_fat_g or 0.0), limits.sat_fat_cap_g, remain["sat_fat_g"])
    if "diabetes" in profile.conditions:
        if sugar_flag:
            flags.append(sugar_flag)
            reasons.append(f"You already had {totals.sugar_g} g sugar today. Adding {food.sugar_g} g from {food.name} crosses the safe daily sugar limit ({limits.sugar_limit_g} g) for diabetes.")
        if "sweet" in food.tags or "refined_carb" in food.tags or (food.carbs_g > 40 and food.fiber_g < 3):
            flags.append("Warning")
            reasons.append(f"{food.name} is high in quick-digesting carbs and low in fiber, which can spike blood sugar.")
    if "hypertension" in profile.conditions:
        if sodium_flag:
            flags.append(sodium_flag)
            reasons.append(f"You already had {totals.sodium_mg} mg salt today. {food.name} adds {food.sodium_mg} mg; your daily limit is {limits.sodium_limit_mg} mg for blood pressure control.")
        if "salty" in food.tags or "fried" in food.tags or food.sodium_mg > 500:
            flags.append("Warning")
            reasons.append(f"{food.name} seems salty/processed. Too much salt can raise blood pressure.")
    if "obesity" in profile.conditions or profile.bmi >= 30:
        if calorie_flag:
            flags.append(calorie_flag)
            reasons.append(f"Your target for today is {limits.calorie_target} kcal. You already had {totals.calories} kcal; {food.name} adds {food.calories} kcal.")
        if "fried" in food.tags or food.fat_g >= 20:
            flags.append("Warning")
            reasons.append(f"{food.name} is oily/fried, which adds many calories with little fullness.")
    if "dyslipidemia" in profile.conditions:
        if sat_fat_flag:
            flags.append(sat_fat_flag)
            reasons.append(f"Your saturated fat cap is {limits.sat_fat_cap_g} g; you've had {totals.sat_fat_g} g already. {food.name} adds {food.sat_fat_g or 0.0} g.")
    if remain["protein_g"] > 0 and food.protein_g >= 15:
        reasons.append(f"Good protein choice — helps with fullness and muscle support (need ~{remain['protein_g']} g more today).")
    if remain["fiber_g"] > 0 and food.fiber_g >= 5:
        reasons.append(f"Nice fiber boost — supports digestion and steadier sugars (need ~{remain['fiber_g']} g more today).")
    if profile.sleep_hours is not None and profile.sleep_hours < 6 and food.sugar_g >= 10:
        flags.append("Warning")
        reasons.append("Poor sleep makes it harder to control sugar cravings — consider a lower-sugar option today.")
    if current_time_is_late(now) and (food.carbs_g >= 40 or food.fat_g >= 20):
        flags.append("Warning")
        reasons.append("It's late — heavy carbs or oily food at night can affect digestion and next-day sugars.")
    final_category = "Acceptable"
    if "Not Recommended" in flags:
        final_category = "Not Recommended"
    elif "Warning" in flags:
        final_category = "Warning"
    tips: List[str] = []
    if final_category != "Acceptable":
        portion_cut = "Try half the portion" if food.grams and food.grams >= 150 else "Choose a smaller portion"
        tips.append(portion_cut)
        if food.sugar_g >= 10:
            tips.append("Pair with protein/fiber (nuts, salad) to slow sugar rise")
        if food.sodium_mg >= 400:
            tips.append("Balance with water and fresh fruits/veggies today")
        if food.fat_g >= 15:
            tips.append("Prefer grilled/steamed over fried today")
    else:
        tips.append("Good choice for today — keep portions sensible")
        if remain["protein_g"] > 0 and food.protein_g < 10:
            tips.append("Consider adding a protein side (dal/curd/eggs)")
    alternatives = ALTERNATIVES_MAP.get(food.name, []).copy()
    for tag in food.tags:
        if tag in TAG_ALTERNATIVES:
            for alt in TAG_ALTERNATIVES[tag]:
                if alt not in alternatives:
                    alternatives.append(alt)
    if profile.vegetarian:
        veg_keywords = ["Paneer", "Dhokla", "Salad", "Sprouts", "Dal", "Roti", "Vegetable", "Chaas", "Buttermilk", "Yogurt", "Idli", "Sandesh", "Rasgulla"]
        def is_veg(a: str) -> bool:
            nonveg_words = ["chicken", "mutton", "fish", "egg"]
            if any(w in a.lower() for w in nonveg_words):
                return False
            if any(kw.lower() in a.lower() for kw in veg_keywords):
                return True
            return True
        alternatives = [a for a in alternatives if is_veg(a)]
    if profile.allergies:
        def safe_alt(a: str) -> bool:
            for allergen in profile.allergies:
                if allergen.lower() in a.lower():
                    return False
            return True
        alternatives = [a for a in alternatives if safe_alt(a)]
    if not reasons:
        reasons.append("This fits your needs today.")
    alternatives = alternatives[:5]
    tips = tips[:5]
    return Evaluation(
        category=final_category,
        reasons=reasons,
        numbers=numbers,
        alternatives=alternatives,
        tips=tips,
    )
# ================= ENHANCEMENTS =================
def pretty_summary(evaluation: Evaluation) -> str:
    """Return a clean, human-readable summary of the evaluation."""
    lines = []
    lines.append(f"Category: {evaluation.category}")
    if evaluation.reasons:
        lines.append("\\nReasons:")
        for r in evaluation.reasons:
            lines.append(f"- {r}")
    if evaluation.tips:
        lines.append("\\nTips:")
        for t in evaluation.tips:
            lines.append(f"- {t}")
    if evaluation.alternatives:
        lines.append("\\nAlternatives:")
        for a in evaluation.alternatives:
            lines.append(f"- {a}")
    return "\n".join(lines)
# Configurable thresholds for warnings
THRESHOLDS = {
    "warning_band": BAND_WARNING,   # Default: 0.8 (80%)
    "notrec_band": BAND_NOTREC      # Default: 1.0 (100%)
}
def set_thresholds(warning: float = 0.8, notrec: float = 1.0):
    """Dynamically adjust thresholds for warnings and not recommended."""
    THRESHOLDS["warning_band"] = warning
    THRESHOLDS["notrec_band"] = notrec
# =================================================
# =================== STRICT / ACTIVITY-AWARE ENHANCEMENTS ===================
def activity_score_to_multiplier(score: int) -> float:
    """
    Map an activity score (1-100) to a small multiplier that allows slot-level flexibility.
    1 -> ~0.92 (very low activity), 50 -> 1.0 (neutral), 100 -> ~1.20 (very active)
    """
    try:
        score = int(score)
    except Exception:
        score = 50
    score = max(1, min(100, score))
    return 0.92 + (score - 1) * (1.2 - 0.92) / 99.0
def _is_nonveg_name(name: str) -> bool:
    nonveg_kw = ["chicken", "mutton", "fish", "egg", "prawn", "shrimp", "meat", "keema", "fish", "egg"]
    return any(k in name.lower() for k in nonveg_kw)
def _is_likely_acceptable_alt(alt: str, profile: UserProfile) -> bool:
    """
    Heuristic-based filter: returns True if an alternative is likely acceptable given profile.
    """
    alt_l = alt.lower()
    # Immediately reject if contains user's allergy
    for a in profile.allergies:
        if a.lower() in alt_l:
            return False
    # Vegetarian filter
    if profile.vegetarian and _is_nonveg_name(alt):
        return False
    good_keywords = [
        "no sugar", "unsweet", "unsweetened", "steamed", "grilled", "roasted", "salad",
        "sprout", "buttermilk", "chaas", "dahi", "curd", "raita", "idli", "dhokla",
        "phulka", "roti", "multigrain", "brown rice", "quinoa", "millet", "sprouts", "dal"
    ]
    bad_keywords = ["sugar", "fried", "jalebi", "gulab", "gulab jamun", "thick shake", "pakora"]
    if any(b in alt_l for b in bad_keywords) and not any(g in alt_l for g in good_keywords):
        return False
    if any(g in alt_l for g in good_keywords):
        return True
    # default to True unless explicitly excluded
    return True
def _format_numerical_snapshot(limits: PersonalLimits, totals: DailyTotals, adj_extra: dict) -> dict:
    """
    Prepare a compact numerical snapshot for UI/JSON consumption.
    adj_extra contains adjustments derived from activity score.
    """
    snapshot = {
        "calorie_target": limits.calorie_target,
        "today_calories": totals.calories,
        "remaining_calories": max(0.0, round(limits.calorie_target - totals.calories, 1)),
        "adjusted_remaining_calories": round(max(0.0, (limits.calorie_target - totals.calories) + adj_extra.get("calories", 0.0)), 1),
        "protein_target_g": limits.protein_target_g,
        "today_protein_g": totals.protein_g,
        "remaining_protein_g": max(0.0, round(limits.protein_target_g - totals.protein_g, 1)),
        "adjusted_remaining_protein_g": round(max(0.0, (limits.protein_target_g - totals.protein_g) + adj_extra.get("protein_g", 0.0)), 1),
        "sugar_limit_g": limits.sugar_limit_g,
        "today_sugar_g": totals.sugar_g,
        "remaining_sugar_g": max(0.0, round(limits.sugar_limit_g - totals.sugar_g, 1)),
        "adjusted_remaining_sugar_g": round(max(0.0, (limits.sugar_limit_g - totals.sugar_g) + adj_extra.get("sugar_g", 0.0)), 1),
    }
    return snapshot
def evaluate(
    food: FoodItem,
    profile: UserProfile,
    today_log: List[FoodItem],
    now: Optional[datetime] = None,
    activity_score: Optional[int] = None,
    slot: Optional[str] = None,
) -> Evaluation:
    """
    Enhanced evaluate() that:
    - Accepts slot-level activity_score (1-100) to allow flexible allowance when user is active
    - Performs strict conditional checks (allergies, vegetarian preference, sodium/sugar/protein/fiber)
    - Produces richer reasoning and classifies alternatives as 'likely acceptable' when possible
    - Returns structured Evaluation compatible with the rest of the system
    """
    now = now or datetime.now()
    limits = derive_personal_limits(profile)
    totals = sum_daily_totals(today_log)
    numbers = {
        "bmi": profile.bmi,
        "calorie_target": limits.calorie_target,
        "protein_target_g": limits.protein_target_g,
        "sugar_limit_g": limits.sugar_limit_g,
        "sodium_limit_mg": limits.sodium_limit_mg,
        "fiber_target_g": limits.fiber_target_g,
        "fat_cap_g": limits.fat_cap_g,
        "sat_fat_cap_g": limits.sat_fat_cap_g,
        "trans_fat_cap_g": limits.trans_fat_cap_g,
        "cholesterol_cap_mg": limits.cholesterol_cap_mg,
        "today_so_far": {
            "calories": totals.calories,
            "sugar_g": totals.sugar_g,
            "sodium_mg": totals.sodium_mg,
            "protein_g": totals.protein_g,
            "fiber_g": totals.fiber_g,
            "fat_g": totals.fat_g,
            "sat_fat_g": totals.sat_fat_g,
        },
        "this_food": {
            "name": food.name,
            "grams": food.grams,
            "calories": food.calories,
            "sugar_g": food.sugar_g,
            "sodium_mg": food.sodium_mg,
            "protein_g": food.protein_g,
            "fiber_g": food.fiber_g,
            "fat_g": food.fat_g,
            "sat_fat_g": food.sat_fat_g or 0.0,
        },
    }
    remain = {
        "calories": _remaining(totals.calories, limits.calorie_target),
        "sugar_g": _remaining(totals.sugar_g, limits.sugar_limit_g),
        "sodium_mg": _remaining(totals.sodium_mg, limits.sodium_limit_mg),
        "fat_g": _remaining(totals.fat_g, limits.fat_cap_g),
        "sat_fat_g": _remaining(totals.sat_fat_g, limits.sat_fat_cap_g),
        "fiber_g": max(0.0, round(limits.fiber_target_g - totals.fiber_g, 1)),
        "protein_g": max(0.0, round(limits.protein_target_g - totals.protein_g, 1)),
    }
    # Activity-based adjustments (slot-level flexibility)
    slot_multiplier = 1.0
    adj_extra = {"calories": 0.0, "protein_g": 0.0, "sugar_g": 0.0}
    if activity_score is not None:
        slot_multiplier = activity_score_to_multiplier(activity_score)
        # Allow a portion (20-30%) of the activity-driven extra calories to be used for this slot
        slot_extra_fraction = 0.28
        adj_extra["calories"] = max(0.0, (slot_multiplier - 1.0) * limits.calorie_target * slot_extra_fraction)
        adj_extra["protein_g"] = max(0.0, (slot_multiplier - 1.0) * limits.protein_target_g * 0.20)
        # For sugar, be conservative: do not increase sugar allowance for activity
        adj_extra["sugar_g"] = 0.0
    # Adjusted remaining values used for per-upload classification
    adj_remain = {
        "calories": max(0.0, remain["calories"] + adj_extra["calories"]),
        "protein_g": max(0.0, remain["protein_g"] + adj_extra["protein_g"]),
        "sugar_g": max(0.0, remain["sugar_g"] + adj_extra["sugar_g"]),
        "sodium_mg": remain["sodium_mg"],
        "fat_g": remain["fat_g"],
        "sat_fat_g": remain["sat_fat_g"],
        "fiber_g": remain["fiber_g"],
    }
    # Quick reject: allergy / vegetarian preference
    reasons: List[str] = []
    flags: List[str] = []
    # Allergen check
    if profile.allergies:
        for allergen in profile.allergies:
            if allergen.lower() in food.name.lower() or any(allergen.lower() in t.lower() for t in food.tags):
                flags.append("Not Recommended")
                reasons.append(f"Contains allergen: {allergen}. Avoid due to allergy.")
                # Suggest safe alternatives
                alternatives = [a for a in ALTERNATIVES_MAP.get(food.name, []) if _is_likely_acceptable_alt(a, profile)]
                if not alternatives:
                    # Fallback tag-based alternatives
                    for tag in food.tags:
                        alternatives += [a for a in TAG_ALTERNATIVES.get(tag, []) if _is_likely_acceptable_alt(a, profile)]
                numbers.update({"slot_multiplier": round(slot_multiplier, 2)})
                numsnap = _format_numerical_snapshot(limits, totals, adj_extra)
                numbers["snapshot"] = numsnap
                tips = ["Avoid this item due to allergy."]
                return Evaluation(category="Not Recommended", reasons=reasons, numbers=numbers, alternatives=alternatives[:5], tips=tips)
    # Vegetarian preference
    if profile.vegetarian and _is_nonveg_name(food.name):
        flags.append("Not Recommended")
        reasons.append("Item appears non-vegetarian while user's profile is vegetarian.")
    
    # Macronutrient-based classification using adjusted remaining allowances
    def classify(metric: str, add_value: float, limit: float, remaining_adj: float) -> Optional[str]:
        if limit <= 0:
            return None
        if remaining_adj <= 0 and add_value > 0:
            return "Not Recommended"
        # ratio: proportion of daily limit that would be used after consuming this food
        ratio = (limit - remaining_adj + add_value) / limit
        if ratio >= THRESHOLDS["notrec_band"]:
            return "Not Recommended"
        if ratio >= THRESHOLDS["warning_band"]:
            return "Warning"
        return None
    sugar_flag = classify("sugar_g", food.sugar_g, limits.sugar_limit_g, adj_remain["sugar_g"])
    sodium_flag = classify("sodium_mg", food.sodium_mg, limits.sodium_limit_mg, adj_remain["sodium_mg"])
    calorie_flag = classify("calories", food.calories, limits.calorie_target, adj_remain["calories"])
    fat_flag = classify("fat_g", food.fat_g, limits.fat_cap_g, adj_remain["fat_g"])
    sat_fat_flag = classify("sat_fat_g", (food.sat_fat_g or 0.0), limits.sat_fat_cap_g, adj_remain["sat_fat_g"])
    # Add flags & reasons based on medical conditions and heuristics
    # Stronger check for diabetes: refined carbs + low fiber -> escalate
    if "diabetes" in profile.conditions:
        if sugar_flag:
            flags.append(sugar_flag)
            reasons.append(f"Diabetes-specific concern: adding {food.sugar_g} g sugar may exceed the safe daily limit ({limits.sugar_limit_g} g).")
        if "sweet" in food.tags or "refined_carb" in food.tags or (food.carbs_g - food.fiber_g > 30):
            # escalate to Not Recommended if it pushes sugar/calories dramatically
            if (food.sugar_g > 15) or (calorie_flag == "Not Recommended"):
                flags.append("Not Recommended")
                reasons.append(f"{food.name} is high in rapid-digesting carbs and may spike blood glucose.")
            else:
                flags.append("Warning")
                reasons.append(f"{food.name} is a refined/high-sugar carbohydrate; consider lower-sugar alternatives.")
    # Hypertension specific checks
    if "hypertension" in profile.conditions:
        if sodium_flag:
            flags.append(sodium_flag)
            reasons.append(f"High sodium alert: {food.name} adds {food.sodium_mg} mg sodium; daily target is {limits.sodium_limit_mg} mg.")
        if "salty" in food.tags or "fried" in food.tags or food.sodium_mg > 500:
            flags.append("Warning")
            reasons.append("This item is salty/processed and may worsen blood pressure control.")
    # Obesity and calorie-centric checks
    if "obesity" in profile.conditions or profile.bmi >= 30:
        if calorie_flag:
            flags.append(calorie_flag)
            reasons.append(f"Calorie concern: today's target {limits.calorie_target} kcal; you've had {totals.calories} kcal so far; this item adds {food.calories} kcal.")
        if "fried" in food.tags or food.fat_g >= 20:
            flags.append("Warning")
            reasons.append("Oily/fried items are energy-dense and contribute to weight gain if frequent.")
    # Dyslipidemia / saturated fat
    if "dyslipidemia" in profile.conditions:
        if sat_fat_flag:
            flags.append(sat_fat_flag)
            reasons.append(f"Saturated fat alert: item adds {food.sat_fat_g or 0.0} g; keep within {limits.sat_fat_cap_g} g.")
    # Sleep-aware sugar sensitivity
    if profile.sleep_hours is not None and profile.sleep_hours < 6 and food.sugar_g >= 12:
        flags.append("Warning")
        reasons.append("Poor sleep can increase sugar cravings and impair glycemic control; prefer lower-sugar choices.")
    # Late night heavy food
    if current_time_is_late(now) and (food.carbs_g >= 40 or food.fat_g >= 20):
        flags.append("Warning")
        reasons.append("Late-night heavy carbs or oily food can impair sleep and glucose control next day.")
    # Positive signals: protein & fiber
    if adj_remain["protein_g"] > 0 and food.protein_g >= 12:
        reasons.append(f"Good protein: provides {food.protein_g} g protein; may help fullness (need ~{adj_remain['protein_g']} g more today).")
    if adj_remain["fiber_g"] > 0 and food.fiber_g >= 4:
        reasons.append(f"Fiber benefit: {food.fiber_g} g fiber helps digestion and moderates sugar absorption (need ~{adj_remain['fiber_g']} g more today).")
    # Final decision assembly
    final_category = "Acceptable"
    if "Not Recommended" in flags:
        final_category = "Not Recommended"
    elif "Warning" in flags:
        final_category = "Warning"
    # Build tips
    tips: List[str] = []
    if final_category != "Acceptable":
        portion_cut = "Try half the portion" if food.grams and food.grams >= 150 else "Choose a smaller portion"
        tips.append(portion_cut)
        if food.sugar_g >= 10:
            tips.append("Pair with protein/fiber (e.g., nuts, curd, salad) to slow sugar rise.")
        if food.sodium_mg >= 400:
            tips.append("Balance with water and fresh fruits/veggies; avoid further salty snacks today.")
        if food.fat_g >= 15:
            tips.append("Prefer grilled/steamed vs fried; opt for a salad side.")
    else:
        tips.append("Suitable choice for now — watch portions and pair with vegetables or protein if needed.")
        if adj_remain["protein_g"] > 0 and food.protein_g < 12:
            tips.append("Consider adding a protein side (dal/curd/eggs) to improve satiety.")
    # Alternatives: prefer ones likely acceptable
    alternatives = ALTERNATIVES_MAP.get(food.name, []).copy()
    for tag in food.tags:
        for alt in TAG_ALTERNATIVES.get(tag, []):
            if alt not in alternatives:
                alternatives.append(alt)
    # Filter alternatives by heuristics and profile
    filtered_alts = [a for a in alternatives if _is_likely_acceptable_alt(a, profile)]
    # As extra step, if original alternatives empty, attempt smart generic list based on tags
    if not filtered_alts:
        # fallback to tag-based suggestions
        filtered_alts = [a for tag in food.tags for a in TAG_ALTERNATIVES.get(tag, [])]
        filtered_alts = [a for a in filtered_alts if _is_likely_acceptable_alt(a, profile)]
    # Provide an explicit explanation if Acceptable: why it is acceptable
    if final_category == "Acceptable":
        accept_reasons = []
        if food.calories <= adj_remain["calories"]:
            accept_reasons.append(f"Fits within adjusted calorie allowance (~{adj_remain['calories']} kcal available).")
        if food.sugar_g <= adj_remain["sugar_g"]:
            accept_reasons.append(f"Sugar content ({food.sugar_g} g) fits within remaining sugar allowance ({adj_remain['sugar_g']} g).")
        if food.sodium_mg <= limits.sodium_limit_mg * 0.6:
            accept_reasons.append("Moderate sodium content for the day.")
        if food.fiber_g >= 3:
            accept_reasons.append("Provides meaningful fiber.")
        if accept_reasons:
            reasons.insert(0, "Acceptable because: " + " ".join(accept_reasons))
    numbers.update({"slot_multiplier": round(slot_multiplier, 2)})
    numbers["adjusted_snapshot"] = _format_numerical_snapshot(limits, totals, adj_extra)
    # Ensure there is at least one reason
    if not reasons:
        reasons.append("This choice aligns with your profile and today's allowances.")
    tips = tips[:5]
    filtered_alts = filtered_alts[:5]
    return Evaluation(
        category=final_category,
        reasons=reasons,
        numbers=numbers,
        alternatives=filtered_alts,
        tips=tips,
    )
# =================== END ENHANCEMENTS ===================
if __name__ == "__main__":
    profile = UserProfile(
        user_id="u1",
        age=45,
        sex="male",
        height_cm=170,
        weight_kg=82,
        activity_level="low",
        sleep_hours=5.5,
        conditions=["diabetes", "hypertension"],
        vegetarian=True,
        allergies=["peanut"]
    )
    today_items = [
        FoodItem(name="Poha", grams=200, calories=250, carbs_g=45, protein_g=6, fat_g=5, sugar_g=4, fiber_g=3, sodium_mg=350, tags=["refined_carb"]),
        FoodItem(name="Dal", grams=150, calories=180, carbs_g=25, protein_g=12, fat_g=4, sugar_g=2, fiber_g=5, sodium_mg=300, tags=[]),
        FoodItem(name="Roti", grams=60, calories=180, carbs_g=30, protein_g=6, fat_g=3, sugar_g=2, fiber_g=4, sodium_mg=150, tags=[]),
    ]
    food = FoodItem(
        name="Gulab Jamun",
        grams=100,
        calories=300,
        carbs_g=50,
        protein_g=4,
        fat_g=12,
        sugar_g=30,
        fiber_g=1,
        sodium_mg=150,
        sat_fat_g=6,
        tags=["sweet"]
    )
    res = evaluate(food, profile, today_items)
    import json
    print(json.dumps(res.as_dict(), indent=2))
