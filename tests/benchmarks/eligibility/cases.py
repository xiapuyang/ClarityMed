"""Curated complaints for the eligibility A/B benchmark.

100 complaints (50 EN + 50 ZH), each labeled with whether the DDXPlus
eligibility check should fire (``expected_eligible``).

Selection criteria:

* **In-scope (25 EN + 25 ZH)**: complaints that name DDXPlus-tracked
  symptoms with clinical specificity. Spread across the dataset's
  categories — cardiopulmonary, respiratory/infectious, GI, neuro,
  allergy/derma — 5 per category to keep per-group recall measurable.
* **Out-of-scope (25 EN + 25 ZH)**: complaints designed to trip false
  positives — knowledge questions, lifestyle attributions, past /
  resolved symptoms, third-party subjects, vague affective text,
  mental health / chronic care, lab queries.

The EN and ZH halves are *not* paired translations. Each language's
recall must reflect what real users typing in that language actually
say, not a translation artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CaseLang = Literal["en", "zh"]
CaseCategory = Literal[
    "cardio",
    "respiratory",
    "gi",
    "neuro",
    "allergy_derma",
    "knowledge_question",
    "lifestyle",
    "past_resolved",
    "third_party",
    "vague_affective",
    "mental_chronic",
    "lab_admin",
]


@dataclass(frozen=True)
class EligibilityCase:
    """One complaint + ground-truth eligibility label.

    Args:
        name: Stable id used for filtering (``--cases name1,name2``)
            and per-trial row joining.
        lang: ``"en"`` or ``"zh"``. Drives which language the
            strategy's ``check`` is called with.
        complaint: The free-text complaint as a real user would type
            it. No "translated" feel — write what a native speaker
            would actually send.
        expected_eligible: Ground-truth label. ``True`` for in-scope
            complaints (DDXPlus eligibility should fire), ``False``
            for everything else.
        category: One of :data:`CaseCategory` — used to group rows in
            the summary CSV and surface per-category recall drops.
    """

    name: str
    lang: CaseLang
    complaint: str
    expected_eligible: bool
    category: CaseCategory


# --- EN in-scope (25) -------------------------------------------------------

_EN_IN_SCOPE: list[EligibilityCase] = [
    # Cardiopulmonary (5)
    EligibilityCase(
        "en_cardio_1",
        "en",
        "I have crushing chest pain that radiates to my left arm, started two hours ago and won't go away.",
        True,
        "cardio",
    ),
    EligibilityCase(
        "en_cardio_2",
        "en",
        "Sudden sharp chest pain when I breathe deep, started after a coughing fit. Hurts to take a full breath.",
        True,
        "cardio",
    ),
    EligibilityCase(
        "en_cardio_3",
        "en",
        "Severe chest pain and shortness of breath that started during a long flight back from Europe.",
        True,
        "cardio",
    ),
    EligibilityCase(
        "en_cardio_4",
        "en",
        "Burning chest pain that gets worse when I lie down, comes on after meals, and antacids barely help.",
        True,
        "cardio",
    ),
    EligibilityCase(
        "en_cardio_5",
        "en",
        "Chest tightness with palpitations and lightheadedness, started this morning and lasted twenty minutes.",
        True,
        "cardio",
    ),
    # Respiratory / infectious (5)
    EligibilityCase(
        "en_resp_1",
        "en",
        "Productive cough with thick green sputum for four days, fever of 39C, and pain in my right chest.",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "en_resp_2",
        "en",
        "Dry cough, fever, body aches and loss of smell for three days. My wife tested positive for COVID yesterday.",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "en_resp_3",
        "en",
        "Wheezing and shortness of breath that my rescue inhaler isn't fixing, getting worse over the last hour.",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "en_resp_4",
        "en",
        "Sore throat with fever, swollen tonsils with white patches, and tender swollen lymph nodes for five days.",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "en_resp_5",
        "en",
        "Persistent cough with night sweats and unintentional weight loss over the last six weeks.",
        True,
        "respiratory",
    ),
    # GI (5)
    EligibilityCase(
        "en_gi_1",
        "en",
        "Severe right upper abdominal pain after a fatty meal, with nausea and one episode of vomiting.",
        True,
        "gi",
    ),
    EligibilityCase(
        "en_gi_2",
        "en",
        "Burning epigastric pain and acid taste in my mouth, worse at night and after meals.",
        True,
        "gi",
    ),
    EligibilityCase(
        "en_gi_3",
        "en",
        "Constant abdominal pain radiating to my back, started yesterday after a heavy drinking night.",
        True,
        "gi",
    ),
    EligibilityCase(
        "en_gi_4",
        "en",
        "Crampy abdominal pain with watery diarrhea and vomiting for two days, can't keep fluids down.",
        True,
        "gi",
    ),
    EligibilityCase(
        "en_gi_5",
        "en",
        "Severe abdominal pain that comes in waves, no bowel movement or gas for three days, abdomen feels distended.",
        True,
        "gi",
    ),
    # Neuro (5)
    EligibilityCase(
        "en_neuro_1",
        "en",
        "Severe one-sided throbbing headache with nausea and sensitivity to light, started this morning.",
        True,
        "neuro",
    ),
    EligibilityCase(
        "en_neuro_2",
        "en",
        "Sharp stabbing pain behind my right eye, lasting thirty minutes at a time, with tearing on that side.",
        True,
        "neuro",
    ),
    EligibilityCase(
        "en_neuro_3",
        "en",
        "Headache with fever, stiff neck, and confusion that came on over the last few hours.",
        True,
        "neuro",
    ),
    EligibilityCase(
        "en_neuro_4",
        "en",
        "Sudden worst headache of my life that started an hour ago, like a thunderclap.",
        True,
        "neuro",
    ),
    EligibilityCase(
        "en_neuro_5",
        "en",
        "Headache that wakes me at the same time every night for the past week, with eye watering.",
        True,
        "neuro",
    ),
    # Allergy / derma (5)
    EligibilityCase(
        "en_allergy_1",
        "en",
        "Sudden hives all over my body and lip swelling about ten minutes after eating shrimp.",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "en_allergy_2",
        "en",
        "Throat tightness, difficulty breathing, and dizziness right after a bee sting.",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "en_allergy_3",
        "en",
        "Itchy raised wheals coming and going for the last two days, no clear trigger.",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "en_allergy_4",
        "en",
        "Sudden facial and tongue swelling, difficulty swallowing, started after taking a new blood pressure pill.",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "en_allergy_5",
        "en",
        "Hives and wheezing about an hour after starting an antibiotic this morning.",
        True,
        "allergy_derma",
    ),
]


# --- EN out-of-scope (25) ---------------------------------------------------

_EN_OUT_SCOPE: list[EligibilityCase] = [
    # Knowledge questions (5)
    EligibilityCase(
        "en_know_1",
        "en",
        "What's the difference between systolic and diastolic blood pressure?",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "en_know_2",
        "en",
        "How does aspirin actually work in the body?",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "en_know_3",
        "en",
        "Why don't antibiotics work on viral infections?",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "en_know_4",
        "en",
        "What are the typical stages of cancer staging?",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "en_know_5",
        "en",
        "How do vaccines train the immune system to recognize a pathogen?",
        False,
        "knowledge_question",
    ),
    # Lifestyle / wellness (5)
    EligibilityCase(
        "en_life_1",
        "en",
        "Should I drink more water first thing in the morning?",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "en_life_2",
        "en",
        "Is intermittent fasting actually good for metabolic health?",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "en_life_3",
        "en",
        "How much sleep do I really need at age forty?",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "en_life_4",
        "en",
        "Are vitamin D supplements worth taking if I work indoors?",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "en_life_5",
        "en",
        "What's the best diet for losing belly fat without losing muscle?",
        False,
        "lifestyle",
    ),
    # Past / resolved (3)
    EligibilityCase(
        "en_past_1",
        "en",
        "I had a really bad migraine last month but it's gone now — should I do anything?",
        False,
        "past_resolved",
    ),
    EligibilityCase(
        "en_past_2",
        "en",
        "I broke my arm five years ago. Anything I should be watching for long-term?",
        False,
        "past_resolved",
    ),
    EligibilityCase(
        "en_past_3",
        "en",
        "I had appendicitis ten years ago. Do I need any kind of follow-up?",
        False,
        "past_resolved",
    ),
    # Third party (3)
    EligibilityCase(
        "en_third_1",
        "en",
        "My mom was just diagnosed with stage two breast cancer, what should we expect?",
        False,
        "third_party",
    ),
    EligibilityCase(
        "en_third_2",
        "en",
        "My son has a tiny scrape on his knee from falling off his bike. Should I worry?",
        False,
        "third_party",
    ),
    EligibilityCase(
        "en_third_3",
        "en",
        "My dad is on warfarin. Can he take ibuprofen for a sore back?",
        False,
        "third_party",
    ),
    # Vague / affective (3)
    EligibilityCase(
        "en_vague_1",
        "en",
        "I just feel off today, can't put my finger on it.",
        False,
        "vague_affective",
    ),
    EligibilityCase(
        "en_vague_2",
        "en",
        "Something just doesn't feel right with my body lately.",
        False,
        "vague_affective",
    ),
    EligibilityCase(
        "en_vague_3",
        "en",
        "I'm just not myself this week.",
        False,
        "vague_affective",
    ),
    # Mental / chronic (3)
    EligibilityCase(
        "en_mental_1",
        "en",
        "I've been really anxious about work lately and can't focus during the day.",
        False,
        "mental_chronic",
    ),
    EligibilityCase(
        "en_mental_2",
        "en",
        "My diabetes hasn't been well controlled, last A1c was 8.5. What should I change?",
        False,
        "mental_chronic",
    ),
    EligibilityCase(
        "en_mental_3",
        "en",
        "I think I might have depression, I've been low and unmotivated for months.",
        False,
        "mental_chronic",
    ),
    # Lab / admin (3)
    EligibilityCase(
        "en_lab_1",
        "en",
        "My ALT came back at 50 on my last blood test. What does that mean?",
        False,
        "lab_admin",
    ),
    EligibilityCase(
        "en_lab_2",
        "en",
        "I got a positive Lyme antibody test but I feel fine. Should I be worried?",
        False,
        "lab_admin",
    ),
    EligibilityCase(
        "en_lab_3",
        "en",
        "How do I get my medical records transferred to a new primary care provider?",
        False,
        "lab_admin",
    ),
]


# --- ZH in-scope (25) -------------------------------------------------------

_ZH_IN_SCOPE: list[EligibilityCase] = [
    # 心肺 (5)
    EligibilityCase(
        "zh_cardio_1",
        "zh",
        "胸口压榨样疼痛，放射到左臂，两小时前开始的，一直没缓解。",
        True,
        "cardio",
    ),
    EligibilityCase(
        "zh_cardio_2",
        "zh",
        "深呼吸时胸口尖锐刺痛，咳嗽后突然出现的，吸气都疼。",
        True,
        "cardio",
    ),
    EligibilityCase(
        "zh_cardio_3",
        "zh",
        "长途飞行回来后突然胸痛、气短，腿也有点肿胀。",
        True,
        "cardio",
    ),
    EligibilityCase(
        "zh_cardio_4",
        "zh",
        "饭后躺下胸口烧灼样疼痛加重，吃了胃药也没怎么缓解。",
        True,
        "cardio",
    ),
    EligibilityCase(
        "zh_cardio_5",
        "zh",
        "今早突然胸闷、心慌、头晕，持续了大概二十分钟。",
        True,
        "cardio",
    ),
    # 呼吸 / 感染 (5)
    EligibilityCase(
        "zh_resp_1",
        "zh",
        "咳嗽伴黄绿色浓痰四天，发烧39度，右胸口疼。",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "zh_resp_2",
        "zh",
        "干咳、发烧、全身酸痛、嗅觉减退三天，老婆昨天新冠阳性。",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "zh_resp_3",
        "zh",
        "喘息、气短，急救吸入剂没什么效果，一个小时内越来越严重。",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "zh_resp_4",
        "zh",
        "嗓子疼、发烧，扁桃体肿大有白点，颈部淋巴结肿痛，已经五天了。",
        True,
        "respiratory",
    ),
    EligibilityCase(
        "zh_resp_5",
        "zh",
        "持续咳嗽伴夜间盗汗、体重下降六周，胃口也不好。",
        True,
        "respiratory",
    ),
    # 消化 (5)
    EligibilityCase(
        "zh_gi_1",
        "zh",
        "吃了油腻的火锅后右上腹剧烈疼痛，恶心，吐了一次。",
        True,
        "gi",
    ),
    EligibilityCase(
        "zh_gi_2",
        "zh",
        "上腹部烧灼样疼痛，嘴里反酸，夜里和饭后更严重。",
        True,
        "gi",
    ),
    EligibilityCase(
        "zh_gi_3",
        "zh",
        "腹痛持续不缓解，放射到后背，昨晚大量喝酒后开始的。",
        True,
        "gi",
    ),
    EligibilityCase(
        "zh_gi_4",
        "zh",
        "腹部绞痛伴水样腹泻和呕吐两天，喝水都喝不下。",
        True,
        "gi",
    ),
    EligibilityCase(
        "zh_gi_5",
        "zh",
        "腹部阵发性剧烈疼痛，三天没排便也没排气，肚子胀得很厉害。",
        True,
        "gi",
    ),
    # 神经 (5)
    EligibilityCase(
        "zh_neuro_1",
        "zh",
        "今早开始的剧烈单侧搏动性头痛，伴恶心和怕光。",
        True,
        "neuro",
    ),
    EligibilityCase(
        "zh_neuro_2",
        "zh",
        "右眼后方刀割样疼痛，每次持续半小时，同侧眼睛流泪。",
        True,
        "neuro",
    ),
    EligibilityCase(
        "zh_neuro_3",
        "zh",
        "头痛伴发烧、脖子僵硬、意识有点模糊，今天上午开始的。",
        True,
        "neuro",
    ),
    EligibilityCase(
        "zh_neuro_4",
        "zh",
        "一小时前突然出现这辈子最严重的头痛，像炸雷一样炸开。",
        True,
        "neuro",
    ),
    EligibilityCase(
        "zh_neuro_5",
        "zh",
        "最近一周每天半夜同一时间头痛把我痛醒，眼睛也流泪。",
        True,
        "neuro",
    ),
    # 过敏 / 皮肤 (5)
    EligibilityCase(
        "zh_allergy_1",
        "zh",
        "吃完虾大概十分钟，全身突然起荨麻疹，嘴唇也肿了。",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "zh_allergy_2",
        "zh",
        "被蜜蜂蛰了之后嗓子发紧、呼吸困难、头晕。",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "zh_allergy_3",
        "zh",
        "最近两天反复起红色风团，痒得难受，没有明显诱因。",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "zh_allergy_4",
        "zh",
        "吃了新换的降压药后脸和舌头突然肿了，吞咽都困难。",
        True,
        "allergy_derma",
    ),
    EligibilityCase(
        "zh_allergy_5",
        "zh",
        "今早开始吃抗生素，一小时后起荨麻疹还伴喘息。",
        True,
        "allergy_derma",
    ),
]


# --- ZH out-of-scope (25) ---------------------------------------------------

_ZH_OUT_SCOPE: list[EligibilityCase] = [
    # 知识类问题 (5)
    EligibilityCase(
        "zh_know_1",
        "zh",
        "高压和低压到底有什么区别？",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "zh_know_2",
        "zh",
        "阿司匹林在体内到底是怎么发挥作用的？",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "zh_know_3",
        "zh",
        "为什么抗生素对病毒感染没用？",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "zh_know_4",
        "zh",
        "癌症的分期一般是怎么划分的？",
        False,
        "knowledge_question",
    ),
    EligibilityCase(
        "zh_know_5",
        "zh",
        "疫苗是怎么训练免疫系统识别病原体的？",
        False,
        "knowledge_question",
    ),
    # 生活方式 (5)
    EligibilityCase(
        "zh_life_1",
        "zh",
        "早上起床后是不是应该多喝点水？",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "zh_life_2",
        "zh",
        "间歇性禁食对代谢健康真的有益吗？",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "zh_life_3",
        "zh",
        "我四十岁了，每天到底应该睡多少小时？",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "zh_life_4",
        "zh",
        "我整天在室内工作，需要补维生素D吗？",
        False,
        "lifestyle",
    ),
    EligibilityCase(
        "zh_life_5",
        "zh",
        "想减肚子又不掉肌肉，应该怎么吃？",
        False,
        "lifestyle",
    ),
    # 既往 / 已愈 (3)
    EligibilityCase(
        "zh_past_1",
        "zh",
        "上个月偏头痛发作得很厉害，现在已经好了，需要做什么处理吗？",
        False,
        "past_resolved",
    ),
    EligibilityCase(
        "zh_past_2",
        "zh",
        "五年前摔断过胳膊，长期来看有什么需要注意的吗？",
        False,
        "past_resolved",
    ),
    EligibilityCase(
        "zh_past_3",
        "zh",
        "十年前做过阑尾炎手术，还需要做什么随访吗？",
        False,
        "past_resolved",
    ),
    # 第三方 (3)
    EligibilityCase(
        "zh_third_1",
        "zh",
        "我妈刚被查出二期乳腺癌，我们接下来该做什么准备？",
        False,
        "third_party",
    ),
    EligibilityCase(
        "zh_third_2",
        "zh",
        "我儿子骑车摔了一下，膝盖一个小擦伤，要紧吗？",
        False,
        "third_party",
    ),
    EligibilityCase(
        "zh_third_3",
        "zh",
        "我爸在吃华法林，腰疼能不能吃布洛芬？",
        False,
        "third_party",
    ),
    # 模糊 / 情绪 (3)
    EligibilityCase(
        "zh_vague_1",
        "zh",
        "今天就是觉得不对劲，又说不上来哪里不舒服。",
        False,
        "vague_affective",
    ),
    EligibilityCase(
        "zh_vague_2",
        "zh",
        "最近身体总觉得哪里有点不对，但又没什么具体症状。",
        False,
        "vague_affective",
    ),
    EligibilityCase(
        "zh_vague_3",
        "zh",
        "这一周感觉自己整个人不太对劲。",
        False,
        "vague_affective",
    ),
    # 心理 / 慢病 (3)
    EligibilityCase(
        "zh_mental_1",
        "zh",
        "最近工作压力大，整天焦虑，上班也没法集中精力。",
        False,
        "mental_chronic",
    ),
    EligibilityCase(
        "zh_mental_2",
        "zh",
        "我的糖尿病一直控制得不好，上次糖化是8.5，应该怎么调整？",
        False,
        "mental_chronic",
    ),
    EligibilityCase(
        "zh_mental_3",
        "zh",
        "我怀疑自己得了抑郁症，已经低落、提不起劲好几个月了。",
        False,
        "mental_chronic",
    ),
    # 化验 / 行政 (3)
    EligibilityCase(
        "zh_lab_1",
        "zh",
        "我上次体检ALT是50，这个值是什么意思？",
        False,
        "lab_admin",
    ),
    EligibilityCase(
        "zh_lab_2",
        "zh",
        "我莱姆病抗体阳性，但是没什么不舒服，要紧吗？",
        False,
        "lab_admin",
    ),
    EligibilityCase(
        "zh_lab_3",
        "zh",
        "怎么把我的病历从原来的诊所转到新的家庭医生那里？",
        False,
        "lab_admin",
    ),
]


ALL_CASES: list[EligibilityCase] = (
    _EN_IN_SCOPE + _EN_OUT_SCOPE + _ZH_IN_SCOPE + _ZH_OUT_SCOPE
)


def select_cases(
    *,
    langs: list[CaseLang] | None = None,
    names: list[str] | None = None,
    categories: list[CaseCategory] | None = None,
) -> list[EligibilityCase]:
    """Filter the case list by lang / name / category.

    All filters are AND-composed; passing ``None`` skips that filter.
    Returns cases in their canonical order (EN in-scope → EN out →
    ZH in-scope → ZH out) so a per-row CSV stays readable.
    """
    out = list(ALL_CASES)
    if langs is not None:
        wanted_langs = set(langs)
        out = [c for c in out if c.lang in wanted_langs]
    if names is not None:
        wanted_names = set(names)
        out = [c for c in out if c.name in wanted_names]
    if categories is not None:
        wanted_cats = set(categories)
        out = [c for c in out if c.category in wanted_cats]
    return out
