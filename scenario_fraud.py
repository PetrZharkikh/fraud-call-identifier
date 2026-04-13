"""
Сценарный детектор мошенничества по тексту (после ASR).

1) Нормализация и разбиение на фразы
2) Категории признаков: проблема, давление, легенда, действие, секретность, деньги/техника
3) Опасные сочетания (бонус)
4) Цепочка: проблема, давление, действие, секретность/контроль (бонус)
5) Нарушения «банковского регламента»: запросы, которых настоящий сотрудник не делает (отдельный бонус)
6) Лёгкое подавление типичных легитимных формулировок (legit_adjustment)
7) Итоговый fraud_score; при score >= порога класс 0 (мошенничество)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Sequence, Tuple


def apply_asr_common_fixes(s: str) -> str:
    """Типичные искажения русского ASR на банковских звонках."""
    repl = (
        ("вербанком", "сбербанком"),
        ("вербанка", "сбербанка"),
        ("вербанку", "сбербанку"),
        ("вербанк", "сбербанк"),
        ("звербанком", "сбербанком"),
        ("звербанка", "сбербанка"),
        ("звербанку", "сбербанку"),
        ("звербанк", "сбербанк"),
        ("бербанку", "сбербанку"),
        ("бербанка", "сбербанка"),
        ("бербанк", "сбербанк"),
        # «кодек», «коду» и т.п. рвут комбо «смс + код»
        ("кодек", "код"),
        ("коду ", "код "),
        ("коду.", "код."),
        ("коду,", "код,"),
        ("коду?", "код?"),
        ("коду!", "код!"),
    )
    for a, b in repl:
        s = s.replace(a, b)
    return s


# --- Категории: подстроки в уже нормализованном тексте (нижний регистр, ё как е) ---

PROBLEM = (
    "блокировк",
    "арест",
    "списан",
    "спишут",
    "мошенник",
    "мошеннич",  # мошенничество, мошеннические действия
    "мошенн",  # мошенническ (корень)
    "подозрительн",
    "подозрен",  # подозрение (часто без «подозрительн»)
    "взлом",
    "уголовн",
    "удержан",
    "заблокир",
    "операция отклонена",
    "санкци",
    "против вас возбужд",
    "хищен",  # хищение средств
    "компрометац",
    "неправомерн",  # неправомерные операции
    "приостановлен перевод",
    "попытк",  # попытка перевода / кредита (осторожно: короткий корень)
    "третьи лица",
    "лишились доступа",
    "оперативн",  # в оперативном порядке (силовые / «расследование»)
    "месте регистрации",
    "месту регистрации",
    "зафиксирован",
)

PRESSURE = (
    "срочно",
    "срочн",  # срочном порядке, срочная блокировка
    "немедленно",
    "не кладите трубку",
    "ни с кем не совет",
    "не сообщайте никому",
    "только вы",
    "прямо сейчас",
    "без промедлен",
    "быстрее",
    "не отключайтесь",
)

LEGEND = (
    "сбербанк",
    "банк",
    "тинькоф",
    "следователь",
    "прокуратур",
    "мвд",
    "министерств внутренних дел",
    "главное управление",
    "экономической безопасности",
    "втб",
    "фсб",
    "фнс",
    "налогов",
    "госуслуг",
    "полици",
    "служба безопасности",
    "оператор банка",
    "центральный банк",
    "банк российской федерации",
    "технического отдела",
    "технический отдел",
    "финансовый департамент",
    "домклик",
)

ACTION = (
    "переведите",
    "перевод средств",
    "сделайте перевод",
    "выполнен перевод",
    "попытка перевода",
    "продублировать номер",
    "продублировать",
    "назовите код",
    "называйте код",
    "скажите код",
    "продиктуйте код",
    "укажите код",
    "указывайте код",
    "укажите данный код",
    "указывайте данный код",
    "указывайте мне",
    "укажите мне",
    "установите прилож",
    "скачайте прилож",
    "anydesk",
    "teamviewer",
    "анидеск",
    "откройте ссылку",
    "перейдите по ссылке",
    "подтвердите операц",
    "подтвердите",
    "введите пароль",
    "назовите номер карты",
    "продублируйте номер",
    "срок действия карты",
    "согласие на операц",
    "направляйтесь в отделение",
    "предоставьте голосовой",
    "озвучьте",
    "проговорите",
)

SECRECY = (
    "никому не говорите",
    "никому не сообщайте",
    "никому не называйте",
    "конфиденциальн",
    "тайна",
    "секрет",
    "никто не должен знать",
    "ни при каких условиях",
)

CONTROL = (
    "оставайтесь на линии",
    "ожидайте оператора",
    "только со мной",
    "ни с кем не разговаривайте",
)

MONEY_TECH = (
    "смс",
    " sms",
    " сообщен",  # «смс сообщение» и т.п.; без ведущего пробела слишком широко
    "карта",
    "карты",
    "карте",
    "картой",
    "карту",
    "счет",
    "счёт",
    "лицевого сч",
    "единого лицевого",
    "финансового номера",
    "пин-код",
    "пин код",
    "cvv",
    "cvc",
    "удаленн",
    "удалённ",
    "удаленный доступ",
    "код из",
    "пятизначн",  # пятизначный код (часто в vishing)
    "личн кабинет",
    "личный кабинет",
    "сбербанк онлайн",
    "втб онлайн",
)

# Веса категорий (если есть хотя бы одно вхождение)
CATEGORY_WEIGHTS: Dict[str, float] = {
    "problem": 2.5,
    "pressure": 2.5,
    "legend": 2.0,
    "action": 3.5,
    "secrecy": 2.5,
    "control": 2.0,
    "money_tech": 1.5,
}

# Пары подстрок (обе должны быть в тексте), бонус к счёту
COMBO_BONUSES: Tuple[Tuple[str, str, float], ...] = (
    ("перевод", "код", 4.0),
    ("карта", "код", 4.0),
    ("смс", "код", 4.5),
    ("банк", "переведите", 3.5),
    ("сбербанк", "код", 3.0),
    ("мошенн", "карт", 3.5),
    # «подозрение + карта» без контекста давало FP на легитимных звонках банка
    ("подозрен", "регистрац", 2.5),
    ("домклик", "подозрен", 3.0),
    ("сбербанк", "закрытие", 3.0),
    ("сбербанк", "вклад", 2.5),
    ("следователь", "мвд", 2.5),
    ("продублировать", "номер", 4.0),
    ("роботизированн", "втб", 2.5),
    # «не оставляли» содержит подстроку «оставляли» — нужна устойчивая фраза
    ("кредитной заявки", "которую вы оставляли", 3.0),
    ("внутрибанковского", "кредитной", 2.0),
    ("неправомерн", "протокол", 2.0),
    ("отклонили", "заявк", 2.5),
    ("отрицательный ответ", "заявк", 2.5),
    ("прервали связь", "банк", 2.5),
    ("заявлен", "банк", 2.5),
    ("приостановлен", "перевод", 3.0),
    ("лицевого сч", "перевод", 2.5),
    ("указывайте", "код", 3.0),
    ("установите", "прилож", 4.0),
    ("удален", "доступ", 3.5),
    ("удалён", "доступ", 3.5),
    ("teamviewer", "", 3.0),  # special: second empty = only first
    ("anydesk", "", 3.0),
)

CHAIN_BONUS_STRICT = 5.0
CHAIN_BONUS_RELAXED = 3.5

# Нарушения регламента легитимного банка: оба фрагмента в norm (кроме b=""), бонус к счёту.
# Не используем голое «пин» — слишком много ложных срабатываний в русском тексте.
REGULATION_VIOLATIONS: Tuple[Tuple[str, str, float, str], ...] = (
    # Запретные просьбы в контексте «службы безопасности»
    ("служба безопасности", "код", 5.0, "reg:sec_service+code"),
    ("служба безопасности", "перевед", 5.5, "reg:sec_service+transfer"),
    ("служба безопасности", "смс", 4.5, "reg:sec_service+sms"),
    ("служба безопасности", "удален", 5.0, "reg:sec_service+remote"),
    # Типичные схемы «безопасный счёт» / срочный перевод
    ("безопасн", "счет", 4.5, "reg:safe_account"),
    ("перевед", "безопасн", 5.0, "reg:transfer+safe"),
    # Манипуляция + деньги / код
    ("никому не говорите", "перевед", 4.5, "reg:secrecy+transfer"),
    ("не кладите трубку", "код", 4.0, "reg:hold+code"),
    ("ни с кем не совет", "код", 3.5, "reg:isolate+code"),
    # Секретные реквизиты (узкие шаблоны)
    ("cvv", "", 3.5, "reg:cvv"),
    ("cvc", "", 3.5, "reg:cvc"),
    ("пин-код", "", 3.0, "reg:pin"),
    ("пин код", "", 3.0, "reg:pin_spaced"),
    # Удалённый доступ / софт (дублирует смысл комбо, но явно как «регламент»)
    ("anydesk", "", 3.0, "reg:anydesk"),
    ("teamviewer", "", 3.0, "reg:teamviewer"),
    ("анидеск", "", 3.0, "reg:anidesk"),
)

_CATS: Dict[str, Tuple[str, ...]] = {
    "problem": PROBLEM,
    "pressure": PRESSURE,
    "legend": LEGEND,
    "action": ACTION,
    "secrecy": SECRECY,
    "control": CONTROL,
    "money_tech": MONEY_TECH,
}


def prepare_norm_and_phrases(raw: str, min_phrase_len: int = 6) -> Tuple[str, List[str]]:
    """
    Сначала делим по . ! ? и переводам строки, потом чистим символы в каждой фрагменте.
    Раньше точки стирались до split — весь диалог оказывался одной «фразой», цепочка не работала.
    """
    s = raw.lower().replace("ё", "е")
    s = apply_asr_common_fixes(s)
    chunks = re.split(r"(?<=[.!?])\s+|\n+", s)
    phrases: List[str] = []
    for ch in chunks:
        ch2 = re.sub(r"[^\w\s\d%+\-]", " ", ch, flags=re.UNICODE)
        ch2 = re.sub(r"\s+", " ", ch2).strip()
        if len(ch2) >= min_phrase_len:
            phrases.append(ch2)
    if not phrases:
        ch2 = re.sub(r"[^\w\s\d%+\-]", " ", s, flags=re.UNICODE)
        ch2 = re.sub(r"\s+", " ", ch2).strip()
        if ch2:
            phrases = [ch2]
    norm = " ".join(phrases) if phrases else ""
    return norm, phrases


def _phrase_categories(phrase: str) -> FrozenSet[str]:
    found: set[str] = set()
    for cat, kws in _CATS.items():
        for kw in kws:
            if kw and kw in phrase:
                found.add(cat)
                break
    return frozenset(found)


def _first_phrase_index_for_categories(
    phrases: Sequence[str], need: Sequence[str]
) -> Dict[str, int]:
    """Минимальный индекс фразы, где впервые встречается категория."""
    idx: Dict[str, int] = {}
    for i, ph in enumerate(phrases):
        cats = _phrase_categories(ph)
        for c in cats:
            if c in need and c not in idx:
                idx[c] = i
    return idx


def chain_bonus_for_phrases(phrases: Sequence[str]) -> Tuple[float, str]:
    """
    Строгая цепочка: проблема, давление, действие, затем секретность или контроль.
    Ослабленная: проблема, действие, затем секретность или контроль (часто в разговоре нет слова «срочно»).
    """
    need = ("problem", "pressure", "action", "secrecy", "control")
    idx = _first_phrase_index_for_categories(phrases, need)
    inf = 10**9
    sec = idx.get("secrecy", inf)
    ctl = idx.get("control", inf)
    last_block = min(sec, ctl)
    if last_block >= inf:
        return 0.0, ""

    if "problem" in idx and "pressure" in idx and "action" in idx:
        if idx["problem"] <= idx["pressure"] <= idx["action"] <= last_block:
            return CHAIN_BONUS_STRICT, "strict"

    if "problem" in idx and "action" in idx:
        if idx["problem"] <= idx["action"] <= last_block:
            return CHAIN_BONUS_RELAXED, "relaxed"

    return 0.0, ""


def legit_adjustment(norm: str) -> Tuple[float, List[str]]:
    """
    Снижение балла для типичных легитимных сценариев, которые по тексту похожи на vishing.
    Не заменяет разметку: только уменьшает ложные срабатывания rule-based скорера.
    """
    adj = 0.0
    tags: List[str] = []
    # На реальном «уточните, вы пытались… платёж/перевод» cat+combo может давать ~12;
    # −5.5 оставляло total≈6.5 (FP при thr=6).
    if (
        "сотрудник" in norm
        and "сбербанк" in norm
        and "уточните" in norm
        and "пытались" in norm
    ):
        adj -= 8.0
        tags.append("legit:sber_operation_verify")
    if "аналитического центра тинькофф инвестиц" in norm:
        adj -= 2.5
        tags.append("legit:tinkoff_invest_intro")
    if "звонок поступает от компании" in norm and "кредит" in norm:
        adj -= 2.0
        tags.append("legit:incoming_credit_company")
    return adj, tags


def regulation_violation_bonus(norm: str) -> Tuple[float, List[str]]:
    """
    Дополнительный балл за сочетания, которые противоречат типичному регламенту
    легитимного банковского общения (запрос кодов, переводов на «безопасный» счёт и т.д.).
    """
    total = 0.0
    tags: List[str] = []
    for a, b, w, tag in REGULATION_VIOLATIONS:
        if not b:
            if a in norm:
                total += w
                tags.append(tag)
        else:
            if a in norm and b in norm:
                total += w
                tags.append(tag)
    return total, tags


def combo_score(norm: str) -> Tuple[float, List[str]]:
    total = 0.0
    tags: List[str] = []
    for a, b, w in COMBO_BONUSES:
        if not b:
            if a in norm:
                total += w
                tags.append(f"combo:{a}")
        else:
            if a in norm and b in norm:
                total += w
                tags.append(f"combo:{a}+{b}")
    return total, tags


def category_score(norm: str) -> Tuple[float, Dict[str, bool]]:
    """Сумма весов по категориям (каждая категория не более раза)."""
    active: Dict[str, bool] = {}
    total = 0.0
    for cat, weight in CATEGORY_WEIGHTS.items():
        kws = _CATS[cat]
        if any(kw in norm for kw in kws if kw):
            active[cat] = True
            total += weight
        else:
            active[cat] = False
    return total, active


@dataclass
class ScenarioBreakdown:
    total: float
    category_sum: float
    combo_sum: float
    chain_bonus: float
    chain_ok: bool
    chain_kind: str
    legit_adj: float
    regulation_bonus: float
    categories: Dict[str, bool] = field(default_factory=dict)
    combo_tags: List[str] = field(default_factory=list)
    legit_tags: List[str] = field(default_factory=list)
    regulation_tags: List[str] = field(default_factory=list)
    n_phrases: int = 0


def scenario_score_breakdown(raw_text: str) -> ScenarioBreakdown:
    norm, phrases = prepare_norm_and_phrases(raw_text)
    if not norm:
        return ScenarioBreakdown(
            total=0.0,
            category_sum=0.0,
            combo_sum=0.0,
            chain_bonus=0.0,
            chain_ok=False,
            chain_kind="",
            legit_adj=0.0,
            regulation_bonus=0.0,
            categories={},
            combo_tags=[],
            legit_tags=[],
            regulation_tags=[],
            n_phrases=0,
        )
    cat_sum, cats = category_score(norm)
    cmb_sum, cmb_tags = combo_score(norm)
    ch_bonus, ch_kind = chain_bonus_for_phrases(phrases)
    leg_adj, leg_tags = legit_adjustment(norm)
    reg_bonus, reg_tags = regulation_violation_bonus(norm)
    total = cat_sum + cmb_sum + ch_bonus + leg_adj + reg_bonus
    return ScenarioBreakdown(
        total=total,
        category_sum=cat_sum,
        combo_sum=cmb_sum,
        chain_bonus=ch_bonus,
        chain_ok=ch_bonus > 0,
        chain_kind=ch_kind,
        legit_adj=leg_adj,
        regulation_bonus=reg_bonus,
        categories=cats,
        combo_tags=cmb_tags,
        legit_tags=leg_tags,
        regulation_tags=reg_tags,
        n_phrases=len(phrases),
    )


def label_from_scenario(score: float, threshold: float) -> int:
    return 0 if score >= threshold else 1
