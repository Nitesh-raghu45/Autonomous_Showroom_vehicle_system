"""Rule-based router. Precision over recall: anything doubtful goes to the LLM."""
from __future__ import annotations

import re
from typing import Any

from .. import clock, inventory
from . import extract as X
from . import normalize
from .intents import (
    APPOINTMENTS, BOOK, CANCEL, CLARIFY, CONFIRM, CONTACT_LOG, FIELD_LOOKUP, GREETING, HISTORY, INVENTORY, LLM,
    NEED_CUSTOMER, NOT_IN_RECORD, REFUSE, REGISTER, SELECT_VEHICLE, SQL_QUERY, UPDATE, Route, RouteContext,
)
from .when import parse_when

_WS = re.compile(r"\s+")


def _low(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text) if re.search(r"\[[^\]]*multi-turn[^\]]*\]", text) else text
    return _WS.sub(" ", text.lower().replace("’", "'").replace("_", " ")).strip()


# ---------------------------------------------------------------------------
# security / scope
# ---------------------------------------------------------------------------

_SECURITY = [
    ("prompt_leak", r"ignore (all |any )?(the )?(previous|prior|above|earlier) |disregard (all |your |the )?(previous|prior|instructions|rules)|system prompt|your (instructions|prompt|rules)\b|reveal .*(prompt|instructions)|print .*prompt|developer mode|jailbreak|not bound by|without (any )?(rules|restrictions)|no rules|tell me everything you know"),
    ("system_disclosure", r"source code|\b(what|which) (llm|ai|model|provider|language model)\b.{0,40}\b(you|this|bot|chatbot|assistant|running|powering|using)\b|running on|powered by|are you (gpt|chatgpt|claude|llama|qwen|openai)|api[\s-]?key"),
    ("impersonation", r"act as (the )?(sales ?person|manager|admin|dealer|priya|rahul)|pretend (to be|you are)|you are now|approve (a |my )?discount"),
    ("bulk_data", r"\b(all|every|other|list of) (the )?customers?\b|customers? (in|from) (your|the) (database|system|crm)|your database|dump (the )?(data|database)|show me everyone"),
]
_OOS = [
    ("weather", r"\bweather|\bforecast|\btemperature outside|\brain(ing)?\b"),
    ("smalltalk", r"\bjokes?\b|\briddle|\bsing\b|\bpoem\b|\bstory\b|your favou?rite|\bhow are you\b"),
    ("general_knowledge", r"capital of|prime minister|president of|\bwho won\b|\bcricket|\bipl\b|football|\bnews\b|\bscore\b of|how tall|how old is"),
    ("finance", r"invest|stock market|bitcoin|crypto|share price|mutual fund"),
    ("writing", r"write (me |us )?(a|an|the) (letter|essay|email|poem|story|code|program|script|resignation|cover)|resignation|cover letter|translate|translation|\bin (japanese|french|spanish|german|hindi)\b"),
    ("abuse", r"repeat (the )?(word|phrase|after)|\b\d{3,} times\b|say .* \d{2,} times"),
]
_SEC = [(k, re.compile(p)) for k, p in _SECURITY]
_OOS_RE = [(k, re.compile(p)) for k, p in _OOS]

_CUST_REF = re.compile(r"\bcustomer\s*(?:id|number|no\.?)?\s*#?:?\s*(\d{4,6})\b")
_REG_VERB = re.compile(r"\b(register|sign me up|sign up|signup|new customer|create (an? )?account|new registration|enrol+)\b")
_ALREADY = re.compile(r"\balready (a |an )?(customer|registered|have an account)|\bagain\b")
_POSS = re.compile(r"\b([a-z]+)'s\s+(?:[a-z]+\s+){0,2}(e-?mail|mail|mobile|phone|number|status|quot\w+|appointments?|enquiry|details|id)\b")
_POSS_OK = {"what", "that", "it", "there", "here", "who", "how", "where", "when", "let", "he", "she", "which", "salesperson", "sales",
            "dealer", "showroom", "car", "quote", "quotation", "enquiry", "today", "tomorrow", "this", "everything", "one", "customer", "vehicle", "whats"}
_THIRD = re.compile(r"\b(friend|brother|sister|wife|husband|father|mother|colleague|neighbou?r|cousin|boss|son|daughter|relative|uncle|aunt)(?:'s)?\b.*\b(status|e-?mail|mobile|phone|quot\w+|details|enquiry|number)\b")
_NUM_LOOKUP = re.compile(r"\b(who|whose)\b|registered to|belongs? to|\bowner\b|a customer|customer with|account (of|for)|whose")
_POSS_NAME = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)'s\b|\b(?:of|about|on|for|customer) ([A-Z][a-z]+ [A-Z][a-z]+)\b")
_REPEATED_CHUNK = re.compile(r"^(?:([a-z]{2,4}))\1{2,}[a-z]*$")
_REPEATED_CHAR = re.compile(r"(.)\1{5,}")


def _looks_gibberish(low: str) -> bool:
    words = re.findall(r"[a-z]{3,}", low)
    if len(words) < 3 or "@" in low:
        return False
    noisy = sum(bool(_REPEATED_CHUNK.fullmatch(word) or _REPEATED_CHAR.search(word)) for word in words)
    return noisy >= 2 and not any(re.search(pattern, low) for _, pattern in _OOS)


def _security(text: str, low: str, ctx: RouteContext) -> Route | None:
    for kind, rx in _SEC:
        if rx.search(low):
            return Route(REFUSE, {"kind": kind}, "security")
    low_ne = X.EMAIL_OK.sub(" ", low)                       # "customer10006@example.com" is an email, not a customer reference
    m = _CUST_REF.search(low_ne)
    if m and not (_REG_VERB.search(low) and _ALREADY.search(low)):
        if not ctx.customer_id or m.group(1) != ctx.customer_id:
            return Route(REFUSE, {"kind": "other_customer"}, "cross-tenant id")
    if _THIRD.search(low_ne) and not (_BOOK_VERB.search(low_ne) or _CANCEL_VERB.search(low_ne)):
        return Route(REFUSE, {"kind": "other_customer"}, "third party")
    own_first = set((ctx.customer_name or "").lower().split())
    for mm in _POSS.finditer(low_ne):
        who = mm.group(1)
        if who not in _POSS_OK and who not in own_first:
            return Route(REFUSE, {"kind": "other_customer"}, "possessive other person")
    nums = [n for n in X.find_numbers(low_ne) if len(n) == 10]
    if nums and _NUM_LOOKUP.search(low_ne) and not any(n in ctx.own_numbers for n in nums):
        return Route(REFUSE, {"kind": "other_customer"}, "number lookup")
    cat = inventory.catalog()
    for mm in _POSS_NAME.finditer(text):
        nm = mm.group(1) or mm.group(2)
        if not nm:
            continue
        if cat.find_makes(nm.lower()) or cat.find_models(nm.lower()):
            continue
        if ctx.customer_name and set(nm.lower().split()) <= set(ctx.customer_name.lower().split()):
            continue
        if ctx.customer_name or ctx.customer_bound:
            return Route(REFUSE, {"kind": "other_customer"}, "cross-tenant name")
    return None


# ---------------------------------------------------------------------------
# clause families
# ---------------------------------------------------------------------------

_BOOK_VERB = re.compile(r"\b(book|schedule|reserve|arrange|set up|put me down|fix (me )?(a|an)|get (me )?(a|an) slot|slot for|i'?ll come|come in for|want to come|would like to come|need to submit|want to submit|confirm (a|an|my) (callback|test drive|appointment))\b(?!ed|ing)")
_CANCEL_VERB = re.compile(r"\b(cancel|call off|undo|withdraw|no longer need|can'?t make it)\b")
_RESCHEDULE = re.compile(r"\brebook\b|\breschedul\w*\s+(my|the|an?|it)\b|\bmove my\b|\bshift my\b|\bpostpone\b|change (the )?(date|time) of my")
_UPDATE_VERB = re.compile(r"\b(update|change|correct|edit|modify|replace)\s+(my|the|both|it|email|name|alt|alternate|contact|mobile|phone|address|title|info|details|to)\b|\b(add|remove|delete|set)\s+(an?\s+|my\s+|the\s+)?(alt|alternate|alternative|secondary|email|name|contact|phone|mobile)\b|\bcorrect the spelling\b")
_UPDATE_OBJ = re.compile(r"\b(e-?mail|name|alt\w*|secondary|mobile|phone|number|title|info|details|profile|contact)\b")
_SELECT_VERB = re.compile(r"\b(go with|choose|select|pick|take|like|love)\b")
_STOCK_REF = re.compile(r"\bS-?\d{3,}\b", re.I)

# Shared negation guard: prevents keyword-triggered mutations when the user is
# saying they DON'T want to do something.  Must fire BEFORE any action matcher.
# Only the few words right before the verb (within the same clause) count, so
# "If it's not too late, book me a test drive" still books.
_NEGATION = re.compile(
    r"\b(?:don'?t|dont|do not|does not|doesn'?t|did not|didn'?t|not|never|no need to|"
    r"wouldn'?t|shouldn'?t|won'?t|can'?t|cannot|couldn'?t|isn'?t|aren'?t)\b"
)
# Deliberation: "Should I take S-1010 or S-1011?" asks for advice, it is not a command.
# (Polite requests such as "Can you book ...?" are NOT deliberation.)
_DELIBERATION = re.compile(
    r"\b(?:should i|shall i|should we|do you think|is it (?:better|worth|wise)|would it be better|"
    r"which (?:one )?(?:should|would)|whether (?:to|i should))\b|"
    r"\bS-?\d{3,}\s+or\s+S-?\d{3,}\b"
)
_NEG_WINDOW_WORDS = 5


def _is_negated(text: str, verb_pattern: re.Pattern) -> bool:
    """True if the action verb is negated ("don't pick", "would never choose") or the
    sentence is deliberation ("should I take X or Y?") rather than a command."""
    m = verb_pattern.search(text)
    if not m:
        return False
    if _DELIBERATION.search(text):
        return True
    prefix = re.split(r"[,.;:!?]", text[:m.start()])[-1]
    window = " ".join(prefix.split()[-_NEG_WINDOW_WORDS:])
    return bool(_NEGATION.search(window))
_VEHICLE_DETAILS = re.compile(r"\b(tell me|show me|give me|what(?:'s| is)|details? (?:about|of)|information (?:about|on)|about)\b")
_REGISTER = re.compile(r"\b(register|sign me up|sign up|signup|new customer|create (an? )?account|new registration|enrol+)\b|first time|first-time")
_INV_CUE = re.compile(r"in stock|available|availability|inventory|do you (have|carry|sell|stock|offer)|you (have|carry|stock|sell|offer)\b|show( me)?\b|find me|looking for|\bi\b(?:\s+\w+){0,2}\s+(?:want|need)\b|any\b|list\b|how many|number of|cheapest|most expensive|what .* models|which .* models|come in|comes in|let me know|tell me|what about")
_INV_NOUN = re.compile(r"\b(cars?|vehicles?|suvs?|hatch\w*|sedans?|mpvs?|evs?|models?|makes|brands?|inventory|hybrids?|diesels?|petrols?|cng|electric|seater|colou?rs?|variants?|trims?)\b")
# A price comparison with no parseable amount is ambiguous. Keep it out of the
# deterministic inventory path so malformed wording cannot reuse prior filters.
_UNPARSED_PRICE_COMPARATOR = re.compile(
    r"\b(greater(?:er)?(?:\s+than)?|more than|above|over|at least|starting from|from|"
    r"less than|below|under|up to|upto|not more than|maximum|minimum)\b"
)
_MALFORMED_PRICE_COMPARATOR = re.compile(r"\bgreater(?:er)?\b|\bthen\s+\d")


def _has_ambiguous_price_comparator(text: str) -> bool:
    return bool(
        re.search(r"\bprice|\bcost", text)
        and (
            _MALFORMED_PRICE_COMPARATOR.search(text)
            or (_UNPARSED_PRICE_COMPARATOR.search(text) and X.parse_price(text) == (None, None))
        )
    )
# words that can follow a make name but are NOT a model ("Tata cars under 15 lakhs", "Toyota diesel", ...)
_GENERIC_AFTER_MAKE = set("""cars car vehicles vehicle suv suvs sedan sedans hatchback hatchbacks mpv mpvs models model range lineup
    inventory stock ev evs cng petrol diesel electric hybrid automatic manual new used under below above between over less more
    upto up to with in from for that which having priced price budget available details info all only and or the any some few
    options please now today""".split())
_MODEL_LIST_REQUEST = re.compile(r"\b(tell me|show me|give me|what are|list)\b.*\b(name|names|models?)\b|\b(name|names|models?)\b.*\ball\b")
_INVENTORY_FOLLOWUP = re.compile(r"\b(tell me|show me|give me|what are|list)\b.*\b(these|those|them|the ones)\b")
# Conversational continuation: the customer is browsing and wants more / other results.
# When the session already has active inventory filters, these short phrases should reuse
# them instead of falling through to the LLM (which may fail on such vague input).
_INVENTORY_CONTINUATION = re.compile(
    r"\b(other|more|another|different|else|alternative|additional)\s*"
    r"(option|options|choice|choices|vehicle|vehicles|car|cars|model|models|one|ones|make|makes)?\b|"
    r"\bwhat else\b|\banything else\b|\bshow me more\b|\bany more\b|"
    r"\bdo you have (other|more|any other|anything)\b|"
    r"\bwhat other\b|\bany other\b"
)
_INVENTORY_CONTEXT = re.compile(r"\b(how many|number of|count|in stock|available|inventory|used|new|evs?|cars?|vehicles?)\b")
# phrases that explicitly abandon whatever vehicle/topic was being discussed - carried
# filters (make/model in particular) must NOT survive these, even if this turn's own
# extraction found no make/model of its own.
_INVENTORY_RESET = re.compile(
    r"\b(any other|a different|something else|some other|another (make|model|car|vehicle|brand)|"
    r"forget (that|it|about that)|never ?mind|start over|new search|not (that|this) one|"
    r"different (car|vehicle|make|model|brand)|other (options?|cars?|vehicles?|makes?|models?)|"
    r"(instead of|other than) (that|this|it)|"
    r"(?:don'?t|do not|not)\b(?!\s+(?:you\s+)?know\b)[^.!?]{0,80}\babout\b)"
)
_NEGATED_INVENTORY_SCOPE = re.compile(r"\b(?:don'?t|do not|not)\b(?!\s+(?:you\s+)?know\b)[^.!?]{0,80}\babout\b")
# refers back to the specific vehicles shown last turn ("in these", "among those", "one of
# these", "from the above ones") - as opposed to a fresh search. Used to restrict a ranking
# query (e.g. "lowest cost") to the actual result set shown, not just to re-run the filters.
_AMONG_RESULTS = re.compile(
    r"\b(in|among|from|of|between) (these|those|the above|em)\b|"
    r"\b(which )?one of (these|those)\b|\bout of (these|those)\b"
)
_PERSONAL = re.compile(r"\b(am|was|did|have) i\b|\b(my|mine)\b.*\b(enquiry|quotation|quote|proposal|salesperson|appointment|deal|order|record|status)\b|\bstock id\b|was i (interested|enquir)|\bi (was|am) (interested|enquir)|\b(interested|enquir\w*) (in|about)\b.*\b(originally|again)\b")
_COMPARE = re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b|which is (cheaper|better|costlier)|\bdifference\b")
_ABOUT = re.compile(r"^(help|help me|what can you do|what do you do|what (?:are we|we are) looking for|(?:so )?what is in your mind|who are you|what are you|are you (a )?(bot|human|robot|ai|real)|how can you help( me)?|menu|options)\W*$")
_HUMAN = re.compile(r"\b(human|real person|representative|customer (care|service|support)|talk to (a |someone|somebody)|speak to|complain\w*|escalat\w*|manager)\b")
_ACK = re.compile(r"^(ok(ay)?|fine|cool|alright|got it|k|nice|great|good|no|nope|nah|nothing( special)?|that'?s all|that is all)\W*$")
_WHICH_PICK = re.compile(r"\bwhich (?:car|vehicle|one|model|suv)s?\b")
_ADVICE = re.compile(r"recommend|suggest|\bbest (car|vehicle|for|option)|good (car|vehicle|for|choice)|worth|is the .*(good|reliable|safe|better)|\badvice\b|opinion|\bbetter\b|family of|for a family|tell me about the|about the (kia|tata|hyundai|maruti|honda|toyota|bmw|mg|mahindra)")
_GREET = re.compile(r"^(hi+|hello+|hey+|namaste|good (morning|afternoon|evening)|thanks?( you)?( a lot| so much| very much)?|thank you( so much| very much)?|ok(ay)? thanks?|bye|goodbye)\W*$")
_COMPLIMENT = re.compile(r"^(ok(ay)?[, ]+)?(you'?re?|you are|this is|that'?s?|that is)\s+(doing (well|good|great)|(working (really |very )?(well|good|great|perfect(ly)?)|(really |very )?(good|great|helpful|awesome|amazing|nice|useful|fast|quick)))\W*$|^(good|great|nice|well done|awesome) (job|work|bot)\W*$")
_COMPLIMENT_EXPECTATION = re.compile(r"^(?:this is|that is|that's) (?:exactly )?what i (?:was )?(?:expecting|expected)(?: from you)?\W*$")
_YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok(ay)?|confirm(ed)?|go ahead|please do|do it|proceed|correct|right|that works|yes,? (please|that\b.*works?))\W*$")
_NO = re.compile(r"^(no|nope|nah|never ?mind|don'?t|do not|stop|cancel( that| it)?|leave it)\W*$")
_VEHICLE_INFO = re.compile(
    r"\b(?:tell me|show me|give me|let me know|what do you know|do you know|know)\b.{0,35}\b(?:about|of)\b|"
    r"\b(?:details?|information|everything|all)\s+(?:about|of)\b"
)


def _clauses(low: str) -> list[str]:
    # "and" between two numbers is a range ("between 25 and 35 lakhs"), not a clause break
    parts = re.split(r",|;|\s[-—–]\s|(?<!\d )\band\b(?! \d)|\bthen\b|\balso\b|\bby the way\b|\bbut\b|\bafter that\b|\bif not\b|(?<=[a-z0-9])\.\s+", low)
    return [p.strip() for p in parts if p and p.strip()]


# auxiliaries only count when inverted at the start ("does the creta ...", not "my wife will come")
_ASKS = re.compile(r"^(?:does|do|is|are|can|could|will|would|should|has|have)\b|\b(?:which|what|whats|how|why|where|who)\b|\?")


def _is_unanswered_ask(c: str) -> bool:
    """A clause no rule family claimed that is itself a question/request (so answering
    only the other clauses would silently drop it). Pleasantries and bare slot data
    ("thanks", "at 10am monday") are not."""
    if _GREET.match(c) or _ACK.match(c) or _YES.match(c) or _COMPLIMENT.match(c):
        return False
    words = re.findall(r"[a-z]{2,}", re.sub(r"\d+\s*(am|pm)\b", " ", c))
    return len(words) >= 3 and bool(_ASKS.search(c))


def is_book(c: str) -> bool:
    if _RESCHEDULE.search(c) or _CANCEL_VERB.search(c):
        return False
    if _is_negated(c, _BOOK_VERB):
        return False
    if _BOOK_VERB.search(c):
        return True
    return bool(X.appt_type(c) and re.search(r"\b(on|at|for)\b.*(\d|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", c) and re.search(r"\b(need|want|like|come|submit|make)\b", c))


def is_cancel(c: str) -> bool:
    if _is_negated(c, _CANCEL_VERB):
        return False
    return bool(_CANCEL_VERB.search(c)) and bool(re.search(r"appointment|test[\s-]?drive|call[\s-]?back|slot|booking|meeting|visit|submission|payment|\bit\b|\ball\b|\bid\b|apt|demo|scheduled|\bthat\b", c))


_NUMBER_CHANGE = re.compile(r"\b(updated|new|changed|different|correct)\s+(mobile\s+|phone\s+|contact\s+|primary\s+|alternate\s+|alt\s+)?(number|no\b|mobile)|\b(mobile|phone|number) (has )?changed\b|\bchange (my )?(mobile|phone|number)|\bmy (mobile|phone) (number )?(is|to) (now )?\d")


_FIELD_CHANGE = re.compile(r"\b(new|updated|changed|correct)\s+(e-?mail|name)\b|\b(e-?mail|name)\s*(?:to|:|=)\s*\S|\b(alt|alternate|alternative|secondary)\s*(?:number|mobile|no|phone)?\s*(?:to|:|=|is)?\s*\+?\d{10}")


def is_update(c: str) -> bool:
    if _NUMBER_CHANGE.search(c) and X.find_numbers(c):
        return True
    if _FIELD_CHANGE.search(c) and not _REGISTER.search(c) and (X.find_email(c)[0] or X.find_numbers(c) or re.search(r"name", c)):
        return True
    return bool(_UPDATE_VERB.search(c)) and bool(_UPDATE_OBJ.search(c)) and not re.search(r"\bupdate on\b|any update|is there (an? )?update", c)


def is_register(c: str) -> bool:
    return bool(_REGISTER.search(c))


def is_inventory(c: str) -> bool:
    if _PERSONAL.search(c) or re.search(r"\bstock id\b", c):
        return False
    cat = inventory.catalog()
    has_noun = bool(_INV_NOUN.search(c))
    has_cue = bool(_INV_CUE.search(c))
    has_veh = bool(cat.find_makes(c) or cat.find_models(c))
    if _has_ambiguous_price_comparator(c):
        return False
    if re.search(r"how many (makes|brands)|what (makes|brands)|which (makes|brands)", c):
        return True
    if X.other_brand(c) and not cat.find_makes(c) and (has_cue or has_noun or len(c.split()) <= 3 or re.search(r"\btell me about\b|\bprice\b", c)):
        return True
    if has_noun and has_cue:
        return True
    if (has_noun or has_veh) and X.parse_price(c) != (None, None):
        return True
    if has_veh and (_MODEL_LIST_REQUEST.search(c) or _VEHICLE_INFO.search(c) or re.search(r"in stock|available|availability|do you (have|carry|sell|stock)|you (have|carry|stock|sell)|show|find me|\bmodels?\b|come in|comes in|price of|cost of|how much (is|does|are|for)|\bprice\b|\bnewest\b|\boldest\b|most recent", c)):
        return True
    return False


_SELECT_INFO_ONLY = re.compile(r"\b(?:like|love) (?:to )?(?:know|see|hear|get|have)?\s*(?:more )?(?:details?|info\w*|specs?|about|price|photos?|pictures?)\b")


def is_select_vehicle(c: str) -> bool:
    if _is_negated(c, _SELECT_VERB) or _SELECT_INFO_ONLY.search(c):
        return False
    return bool(_SELECT_VERB.search(c) and _STOCK_REF.search(c))


def _is_filter_only_refinement(low: str) -> bool:
    """True for a message that names no vehicle noun/make/model of its own (price, fuel,
    body style, seats, transmission, new/used, a superlative sort) - it only makes sense
    as a refinement of whatever inventory search is already active this session, e.g.
    "50 lacs above and new" or "only automatic" after asking about a specific model."""
    if (_PERSONAL.search(low) or is_update(low) or is_register(low) or is_book(low)
            or is_cancel(low) or _RESCHEDULE.search(low) or is_select_vehicle(low) or _oos(low)):
        return False
    if _has_ambiguous_price_comparator(low):
        return False
    return bool(X.inventory_filters(low))


# ---- record (read) family ----------------------------------------------------------

_NOT_IN_RECORD = re.compile(
    r"salesperson'?s? (e-?mail|mail|phone|number|mobile|contact)|contact (details|number|info) (of|for) (my |the )?(salesperson|sales)|reduce (the )?price|lower (the )?price|bargain|haggle|price match|best (price|deal)|too expensive|verified|verification|credit score|cibil|\bloan\b|tenure|aadhaar|aadhar|\bpan\b|insurance|relationship manager|discount|negoti|trade[\s-]?in|\bexchange\b|sell my (?:old )?car|(?:value|valuation|worth) of my (?:old |current )?car|my (?:old |current )?car'?s? (?:value|valuation|worth)|warranty|"
    r"home address|postal address|street address|residential|which bank|\bbank\b|financ(e|ing)|previous owner|colou?r (did|do) i|colou?r i (selected|chose|picked)|lifetime spend|total spend|"
    r"kilomet\w+ (have )?i|\bkm i\b|how many times .*(visited|visit)|visited the showroom|\bemi\b|on[\s-]?road|"
    r"refund|complaint status|birthday|\bage\b of me|date of birth|\bdob\b|gender|occupation|income|salary|down ?payment amount"
)
_FIELD_RULES: list[tuple[str, str]] = [
    ("email", r"e-?mail|mail id|mail address"),
    ("alt_mobile", r"\b(alt|alternate|alternative|secondary)\b"),
    ("mobile", r"\b(mobile|phone|cell)\b|registered (number|no)\b|\bmy number\b"),
    ("customer_id", r"customer (id|number|no|code)|\bmy id\b"),
    ("name", r"\bmy name\b|name (on|in) (file|record)|\bnaam\b"),
    ("salesperson", (
        r"sales ?(person|guy|girl|rep|executive|manager|team|man)|salesperson|"
        r"who('s| is)? (handling|assigned|looking)|who should i contact|who.*handling|"
        r"kaun.*(salesman|salesperson)|"
        r"(?:full )?name of (?:the )?(?:person|salesperson).{0,40}"
        r"(?:associated|handling|assigned).{0,30}enquir|"
        r"person.{0,30}associated with (?:this|my|the) enquir"
    )),
    ("enquiry_id", r"enquiry (id|number|no)\b|enquiry reference|reference number|give me the enquiry id"),
    ("age", r"how long ago|how old is my|how many days"),
    ("created", r"when (was|did|is) (my|the|this) enquiry|when did i (first )?enquire|(date|day).*(enquiry|raised|created)|(enquiry|raised|created).*\bdate\b|created on|raised on|when was (it|this) (raised|created)"),
    ("new_or_used", r"new (car |vehicle )?(or|ya) (a )?used|used (car |vehicle )?(or|ya) (a )?new|new or (a )?used|used one"),
    ("vehicle", r"(make|model|car|vehicle).*(interested|enquir|looking|chosen|selected)|(interested|enquir|originally|chosen|selected).*(make|model|car|vehicle)|which (model|make|car)|what car (was|am|did)"),
]
_DELIVERY = r"deliver\w*|car ready|ready for (delivery|pickup)|when (will|can) i (get|receive|take)|when will my car"
_Q_WORDS = re.compile(r"how much (do|will|should) i (owe|pay)|kitna (dena|paisa|amount|payment|total)|quotation|\bquote\b|quoted|proposal|\bvat\b|\btax\b|\bgst\b|grand total|sale (price|amount)|"+_DELIVERY+r"|stock id|reserved|(price|cost) of (the|my) (car|vehicle)")
_QUESTION_FORM = re.compile(r"\b(kya|kaun|kab|kitna|kitne|kitni|batao|bata|btao|dikhao|dikha|bataiye|bataye|what|whats|what's|which|who|when|how|why|tell|give|show|confirm|remind|list|summari[sz]e|explain|describe|compare|any|does)\b|\b(is|are|has|have|was|were|did|do|am) (my|there|the|this|i|you|anyone)\b|\bcan i\b|\bcould i\b|\bwill i\b|let me (know|see|check)|(want|need) to (know|see)|\bknow\b|\?")
_ENQ_WORDS = re.compile(r"enquir|deal|purchase|\bstatus\b|\bstage\b|\bconverted\b|\bwin\b|\bwon\b|\blose\b|\blost\b")
_CONTACT = re.compile(r"contact(ed)? (me|history)|last (interaction|contact|conversation|in touch|showroom visit|call|email|whatsapp)|interactions?|in touch|how many times .*(contacted|called|reached)|contact history|notes (from|do you have)|what was discussed|outcome of my last|who called me|transcript|(when|how) (was|were) i (last )?contacted|was i contacted|did anyone (call|contact|email)")
_APPT_READ = re.compile(r"appointment|scheduled|upcoming|call[\s-]?back|\bdemo\b|test[\s-]?drive|\bvisit\b|next scheduled")
_APPT_CUE = re.compile(r"do i have|any\b|list|show|view|check|see\b|what|when|is my|status|confirmed|remind|next|more than one|coming up|which")
_TIME_WORDS = re.compile(r"last week|this week|yesterday|last month|this month|last year|today\b")


def _fields(low: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[?;\n]+", low) if part.strip()]
    if len(parts) > 1:
        return list(dict.fromkeys(field for part in parts for field in _fields(part)))
    if (re.search(r"\b(?:my|the)\s+(?:last|previous|earlier)\s+enquiry\b", low)
            and not re.search(r"\b(date|when|raised|created|salesperson|status|id)\b", low)):
        return ["enq_summary"]
    q = bool(_Q_WORDS.search(low))
    out: list[str] = []
    if q:
        if re.search(r"proposal (id|number)", low):
            out.append("proposal_id")
        if re.search(r"\bstatus\b|accepted|pending|rejected|expired|sent", low):
            out.append("q_status")
        if re.search(_DELIVERY, low):
            out.append("delivery_delivered" if re.search(r"delivered", low) else "delivery")
        if re.search(r"stock id|reserved", low):
            out.append("stock_id")
        if re.search(r"breakdown|itemi[sz]ed|cost details|break down", low):
            out.append("breakdown")
        if re.search(r"before (vat|tax|gst)|excluding|sale price|sale amount|ex[\s-]?showroom", low) and not re.search(r"plus|\+", low):
            out.append("sale_total")
        if re.search(r"how much (?:is |of )?(?:the )?(?:vat|tax|gst)|(?:vat|tax|gst) (?:amount|(?:is )?included|charged)", low):
            out.append("tax_total")
        if re.search(r"grand total|\btotal\b|including|plus|\+|price quoted|final price|payable|total amount|how much (do|will|should) i (owe|pay)", low):
            out.append("grand_total")
        if re.search(r"what (car|vehicle|make|model)|which (car|vehicle|make|model)|make/model|car .*quot", low):
            out.append("q_vehicle")
        # Also collect customer/enquiry fields when the user combines them with a
        # quotation question (e.g. "my name and my quotation status").
        for name, pattern in _FIELD_RULES:
            if re.search(pattern, low):
                out.append(name)
        if re.search(r"\bstatus\b|converted|\bwin\b|\bwon\b|\blose\b|\blost\b|\bstage\b|still open|open or closed|closed or|in progress", low) and re.search(r"enquir|deal|purchase|stage|open|closed", low):
            out.append("status")
        if not out:
            out.append("q_summary" if len(low.split()) <= 3 else "grand_total")
        return list(dict.fromkeys(out))

    if re.search(r"(full|all|complete) (my )?details|everything about me|all my (info|details)|my profile|about me\b|\bmy (details|info|information|account)\b", low):
        return ["profile_summary"]
    if re.search(r"^(my )?enquiry$|(my|show my|show) enquiry (details|summary|info)|enquiry details", low):
        return ["enq_summary"]
    if re.search(r"summari[sz]e my enquiry|enquiry (summary|in one line)|my enquiry in one", low):
        return ["enq_summary"]
    for name, pat in _FIELD_RULES:
        if re.search(pat, low):
            out.append(name)
    if re.search(r"(any|is there (an? )?)update.*enquiry|enquiry.*(any|since).*update|update on my enquiry", low):
        return ["status", "created"]
    m = re.search(r"what does ['\"]?(open|closed[ _]?won|closed[ _]?lost|closed)['\"]? (status )?mean", low)
    if m:
        return ["status_meaning:" + m.group(1).replace(" ", "_")]
    if "new_or_used" in out and not re.search(r"\b(make|model)\b", low) and "vehicle" in out:
        out.remove("vehicle")
    if "alt_mobile" in out and "mobile" in out and not re.search(r"\b(mobile|phone|number)\b.*\b(and|&)\b.*\b(alt|alternate)|\b(alt|alternate).*\b(and|&)\b", low):
        out.remove("mobile")
    if re.search(r"\bstatus\b|converted|\bwin\b|\bwon\b|\blose\b|\blost\b|\bstage\b|still open|open or closed|closed or|in progress|\bdeal\b", low) and re.search(r"enquir|deal|purchase|status|converted|stage|open|closed", low):
        out.insert(0, "status")
    return list(dict.fromkeys(out))


def _ctype(c: str) -> str | None:
    found = [t for t, p in (("SHOWROOM_VISIT", r"showroom visit|visit"), ("CALL", r"\bcall(ed)?\b|phone call"), ("EMAIL", r"e-?mail"), ("WHATSAPP", r"whats ?app")) if re.search(p, c)]
    return found[0] if len(found) == 1 and re.search(r"last|notes|discussed", c) else None


def _terse(c: str) -> bool:
    """1-5 word fragments like 'status', 'my quotation', 'which salesman' are questions too (no digits/emails = not slot data)."""
    return len(c.split()) <= 5 and not re.search(r"\d|@", c)


def record_route(c: str) -> Route | None:
    """Detect a read-only question about the customer's record."""
    if not _QUESTION_FORM.search(c) and not _terse(c):
        return None
    if _NOT_IN_RECORD.search(c):
        return Route(NOT_IN_RECORD, {"topic": _NOT_IN_RECORD.search(c).group(0)}, "unsupported field")
    if re.search(r"\bwhy\b", c) and re.search(r"enquir|quotation|quote|proposal|closed|lost|rejected|expired|status", c):
        return Route(FIELD_LOOKUP, {"fields": ["why_status"], "q": bool(_Q_WORDS.search(c))}, "why")
    if re.search(r"compare|vs\b|versus|last month.*status|status.*last month", c) and re.search(r"enquir|status|quote", c):
        return Route(NOT_IN_RECORD, {"topic": "history comparison"}, "no history")
    if _CONTACT.search(c):
        if _TIME_WORDS.search(c) and re.search(r"did anyone|who called|how many", c):
            return Route(LLM, {}, "contact window needs date reasoning")
        mode = "count" if re.search(r"how many times", c) else "all" if re.search(r"\ball\b|entire|history|chronolog|list|every", c) else "last"
        return Route(CONTACT_LOG, {"mode": mode, "transcript": "transcript" in c, "ctype": _ctype(c), "want": "type" if re.search(r"how was i contacted|call, email|which (channel|type)", c) else "date" if re.search(r"\bwhen\b", c) else "notes" if re.search(r"notes|discussed|say|outcome", c) else "all"}, "contact log")
    if _APPT_READ.search(c) and _APPT_CUE.search(c) and re.search(r"\b(my|i have|do i|any)\b", c) and not (is_book(c) or is_cancel(c)):
        return Route(APPOINTMENTS, {}, "appointments", needs_appts=True)
    fields = _fields(c)
    if fields:
        return Route(FIELD_LOOKUP, {"fields": fields}, "fields")
    return None


def _oos(low: str) -> str | None:
    for kind, rx in _OOS_RE:
        if rx.search(low):
            return kind
    return None


def _families(low: str, cache_route: dict[str, Route | None], unmatched: list[str] | None = None) -> list[str]:
    fams: list[str] = []
    for c in _clauses(low):
        before = len(fams)
        if is_cancel(c):
            fams.append("cancel")
        elif _RESCHEDULE.search(c):
            fams.append("reschedule")
        elif is_book(c):
            fams.append("book")
        elif is_update(c):
            fams.append("update")
        elif is_select_vehicle(c):
            fams.append("select_vehicle")
        elif is_register(c):
            fams.append("register")
        elif _oos(c):
            fams.append("oos")
        elif is_inventory(c):
            fams.append("inventory")
        else:
            r = record_route(c)
            if r is not None:
                fams.append("record")
                cache_route.setdefault("record", r)
        if unmatched is not None and len(fams) == before and _is_unanswered_ask(c):
            unmatched.append(c)
    return list(dict.fromkeys(fams))


# ---------------------------------------------------------------------------
# handlers (extraction on the FULL text)
# ---------------------------------------------------------------------------

def _h_book(text: str, low: str) -> Route:
    today = clock.today()
    w = parse_when(text, today)
    atype = X.appt_type(low)
    slots: dict[str, Any] = {"appointment_type": atype, "when": w, "notes": X.notes_after(text)}
    if re.search(r"\b(two|three|both|2|3)\b.*(test[\s-]?drives?|appointments?|slots?)|for (me )?and my\b", low):
        return Route(CLARIFY, {"kind": "multi_booking"}, "multi booking")
    cat = inventory.catalog()
    veh = cat.find_models(low)
    if veh and not slots["notes"]:
        slots["notes"] = f"{veh[0][0]} {veh[0][1]}"
    elif veh and slots["notes"]:
        slots["notes"] = f"{veh[0][0]} {veh[0][1]} - {slots['notes']}"
    return Route(BOOK, slots, "book")


def _h_cancel(text: str, low: str) -> Route:
    m = re.search(r"\b(apt[-\s]?\w+|[0-9a-f]{12})\b", text, re.I)
    appt_id = m.group(1).upper().replace(" ", "-") if m else None
    return Route(CANCEL, {
        "appt_id": appt_id,
        "appointment_type": X.appt_type(low) if not appt_id else None,
        "all": bool(re.search(r"\ball\b|every", low)),
    }, "cancel", needs_appts=True)


def _h_update(text: str, low: str) -> Route:
    slots: dict[str, Any] = {"problems": [], "refused": []}
    email, bad_email = X.find_email(text)
    if email:
        slots["email"] = email
    elif bad_email or re.search(r"\b(update|change|set)\b.*\be-?mail\b", low):
        slots["problems"].append("email_invalid" if bad_email else "email_missing")

    nm = X.name_after(text, r"\bname\s+(?:spelling\s+)?(?:to|as|is)\s+") or X.name_after(text, r"\bname\s*[:=]\s*")
    if nm:
        slots["name"] = nm
    elif re.search(r"\b(change|update|correct)\b.*\bname\b", low) and "name" not in slots:
        slots["problems"].append("name_missing")

    if re.search(r"\b(remove|delete|clear)\b.*\b(alt|alternate|alternative|secondary)\b", low):
        slots["clear_alt"] = True
    elif re.search(r"\b(alt|alternate|alternative|secondary)\b", low):
        nums = X.find_numbers(text)
        if nums:
            slots["alt_mobile_no"] = nums[0]
        else:
            slots["problems"].append("alt_missing")

    if re.search(r"\btitle\b", low):
        slots["refused"].append("title")
    elif "alt_mobile_no" not in slots and not slots.get("clear_alt") and (_NUMBER_CHANGE.search(low) or re.search(r"\b(primary|main|registered)\b", low)):
        nums = X.find_numbers(text)
        if re.search(r"\b(primary|main|registered)\b", low):
            if nums:
                slots["mobile_no"] = nums[0]
            else:
                slots["problems"].append("mobile_missing")
        elif nums:
            slots["ambiguous_number"] = nums[0]          # primary or alternate? ask, and remember the number
        else:
            slots["problems"].append("mobile_missing")
    if not any(k in slots for k in ("email", "name", "alt_mobile_no", "mobile_no", "ambiguous_number", "clear_alt")) and not slots["problems"] and not slots["refused"]:
        slots["problems"].append("nothing_specified")
    return Route(UPDATE, slots, "update")


def _h_register(text: str, low: str, ctx: RouteContext) -> Route:
    if _ALREADY.search(low) or ctx.customer_bound and ctx.customer_id:
        return Route(CLARIFY, {"kind": "already_registered"}, "already registered")
    cat = inventory.catalog()
    s: dict[str, Any] = {"problems": []}
    name = (X.name_after(text, r"\b(?:my\s+)?name\s*(?:is|:|=)?\s*", strict=False)
            or X.name_after(text, r"\b(?:register|sign me up|sign up|signup|enrol+|new customer|new registration|create (?:an )?account for)\b\s*(?:me|as|form|here|again)?\s*[:,\-—–]*\s*(?:as\s+)?"))
    if name and len(name.split()) >= 1:
        s["name"] = name
    email, bad = X.find_email(text)
    if email:
        s["email"] = email
    elif bad:
        s["problems"].append("email_invalid")
    nums = [n for n in X.find_numbers(text) if not (len(n) == 5 and n.startswith("1"))]
    tens = [n for n in nums if len(n) == 10]
    if tens:
        s["mobile_no"] = tens[0]
        if len(tens) > 1:
            s["alt_mobile_no"] = tens[1]
    elif nums:
        s["problems"].append("mobile_invalid")
    t = re.search(r"\btitle\s*[:=]?\s*(mr|mrs|ms|miss|dr|mx|prof)\b", low)
    if t:
        s["title"] = t.group(1).capitalize()
    f = X.inventory_filters(text)
    if f.get("models"):
        s["interested_model"] = f["models"][0]
        s["interested_make"] = (f.get("makes") or [None])[0]
    elif f.get("makes"):
        s["interested_make"] = f["makes"][0]
    if f.get("vtype"):
        s["new_or_used"] = f["vtype"]
    return Route(REGISTER, s, "register")


def _h_inventory(text: str, low: str, prior: dict[str, Any] | None = None, result_ids: list[str] | None = None) -> Route:
    f = X.inventory_filters(text)
    ob = X.other_brand(low)
    if ob and not f.get("makes") and not f.get("models"):
        return Route(INVENTORY, {"filters": {"_brand": ob}, "mode": "unknown_brand"}, "unknown brand")

    # A correction such as "I am not asking about Tata" mentions the make while
    # explicitly removing it from scope. Keep unrelated filters from the same
    # sentence, but do not search for the negated make/model.
    if _NEGATED_INVENTORY_SCOPE.search(low):
        f.pop("makes", None)
        f.pop("models", None)

    # ---- carry forward the make/model (and other filters) from the previous turn -------------
    # A refinement like "50 lacs above and new" or "only automatic" names no vehicle of its
    # own; without this, each turn is judged in isolation and the conversation "forgets" what
    # the person was just looking at. We only merge - never let old text override this turn's
    # own words - and we never carry forward across an explicit topic change.
    reset = _INVENTORY_RESET.search(low)
    if prior and not reset:
        base = {k: v for k, v in prior.items() if k != "_brand"}
        # A new make/model/body query starts a fresh scope; it must not inherit a
        # price bound from the previous search. Likewise, a new price expression
        # replaces both old bounds ("above 50L" must not retain "under 10L").
        if any(key in f for key in ("makes", "models", "body")):
            for key in ("makes", "models", "body", "min_price", "max_price", "stock_ids", "variants"):
                base.pop(key, None)
        elif "min_price" in f or "max_price" in f:
            base.pop("min_price", None)
            base.pop("max_price", None)
        f = {**base, **f}

    # ---- "in these" / "among those" / "one of these" -------------------------------------
    # Restrict to the literal stock_ids shown last turn, not just to whatever the merged
    # filters happen to re-match (the shown page may be a subset, or ties may have broken
    # differently) - this is what makes "these" refer to something concrete.
    if result_ids and not reset and _AMONG_RESULTS.search(low):
        f = {**f, "stock_ids": result_ids}
    return _inventory_route(f, low)


_DISTINCT_FIELDS = [
    ("variant", re.compile(r"\bvariants\b|\btrims\b|\bvariant (?:options|list|names?)\b")),
    ("fuel", re.compile(r"\b(?:what|which) fuels?\b|\bfuel (?:options?|types?|choices?)\b|\bengine options?\b")),
    ("transmission", re.compile(r"\btransmission (?:options?|types?|choices?)\b|\b(?:what|which) transmissions?\b")),
    ("year", re.compile(r"\b(?:what|which) (?:model )?years?\b|\bmodel years?\b|\bmanufactur\w* years?\b")),
    ("body", re.compile(r"\bbody (?:types?|styles?)\b|\b(?:what|which) body\b")),
]


def _inventory_route(f: dict[str, Any], low: str) -> Route:
    extra: dict[str, Any] = {}
    if re.search(r"how many (makes|brands)|number of (makes|brands)|which (makes|brands)|what (makes|brands)", low):
        mode = "makes"
    elif re.search(r"colou?rs?", low):
        mode = "colors"
    elif next((fld for fld, rx in _DISTINCT_FIELDS if rx.search(low)), None):
        mode = "distinct"
        extra["field"] = next(fld for fld, rx in _DISTINCT_FIELDS if rx.search(low))
    elif re.search(r"what .*models|which .*models|models (do|are)", low) or _MODEL_LIST_REQUEST.search(low):
        mode = "models"
    elif re.search(r"how many|number of|\bcount\b|total (number|count)", low):
        mode = "count"
    elif f.get("sort") in ("price_asc", "price_desc", "km_asc", "year_desc", "year_asc") and (re.search(r"cheapest|most expensive|costliest|lowest|highest|least|priciest|most affordable|maximum|minimum|\bmax\b|\bmin\b|newest|oldest|most recent|latest", low) or f.get("limit")):
        mode = "superlative"
    else:
        mode = "list"
    return Route(INVENTORY, {"filters": f, "mode": mode, **extra}, "inventory")


# ---------------------------------------------------------------------------
# structured inventory operations: aggregate / compare / detail
# (deterministic - exact numbers from the in-memory catalogue; the SQL engine and the LLM only
#  see what these patterns cannot express)
# ---------------------------------------------------------------------------

_GROUP_WORDS: list[tuple[str, str]] = [
    ("make", r"makes?|brands?|manufacturers?|compan(?:y|ies)"),
    ("fuel", r"fuels?(?:[\s-]types?)?"),
    ("transmission", r"transmissions?|gearbox(?:es)?"),
    ("body", r"body(?:[\s-](?:types?|styles?))?|segments?"),
    ("year", r"(?:model[\s-])?years?"),
    ("color", r"colou?rs?"),
    ("model", r"models?"),
]
_GROUP_RES: list[tuple[str, re.Pattern[str], re.Pattern[str]]] = [
    (
        fld,
        re.compile(rf"\b(?:each|every|per)\s+(?:{w})\b|\b(?:{w})[\s-]?wise\b|\bby\s+(?:{w})\b|\bgroup(?:ed)?\s+by\s+(?:{w})\b|\bacross\s+(?:all\s+)?(?:{w})\b|\bbreakdown of .*\b(?:{w})\b"),
        re.compile(rf"\b(?:which|what)\s+(?:{w})\s+(?:has|have|offers?|sells?|carries|stocks?|is|are|gives?)\b|\b(?:most|least)\s+(?:common|popular|frequent)\s+(?:{w})\b"),
    )
    for fld, w in _GROUP_WORDS
]
_NEW_VS_USED = re.compile(r"\bnew\s+(?:vs\.?|versus|and|or|to)\s+used\b|\bused\s+(?:vs\.?|versus|and|or|to)\s+new\b")
_AVG_WORD = re.compile(r"\b(?:average|avg|mean)\b")
_MIN_WORD = re.compile(r"\bcheapest\b|\bleast expensive\b|\blowest\b|\bminimum\b|\bleast\b|\bfewest\b|\bsmallest\b|\bmin\b")
_MAX_WORD = re.compile(r"\bmost\b|\bhighest\b|\bmaximum\b|\bmax\b|\blargest\b|\bbiggest\b|\bcostliest\b|\bpriciest\b|\bexpensive\b|\btop\b")


def _agg_metric(low: str) -> str:
    if _AVG_WORD.search(low):
        return "avg_km" if re.search(r"\bmileage\b|\bkms?\b|\bodometer\b", low) else "avg_price"
    if re.search(r"\bcheapest\b|\blowest (?:price|cost|priced)\b|\bleast expensive\b|\bmin(?:imum)? price\b", low):
        return "min_price"
    if re.search(r"\bmost expensive\b|\bcostliest\b|\bpriciest\b|\bhighest (?:price|cost|priced)\b|\bmax(?:imum)? price\b", low):
        return "max_price"
    return "count"


def aggregate_route(text: str, low: str, result_ids: list[str] | None = None) -> Route | None:
    """Group-by / average / price-range questions -> a deterministic INVENTORY route, else None."""
    group: str | None = None
    pick: str | None = None
    for fld, rx, pick_rx in _GROUP_RES:
        if rx.search(low):
            group = fld
        if pick_rx.search(low):
            group, pick = fld, "pick"
        if group:
            break
    if group is None and _NEW_VS_USED.search(low):
        group = "type"
    f = {k: v for k, v in X.inventory_filters(text).items() if k not in ("sort", "limit")}
    if result_ids and _AMONG_RESULTS.search(low):
        f["stock_ids"] = result_ids
    base = {"filters": f, "sticky": False}

    if group:
        metric = _agg_metric(low)
        if pick:
            if metric == "count" and not re.search(r"\bmost\b|\bleast\b|\bfewest\b|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b", low):
                return None                      # "which make is my enquiry for" - not a ranking question
            if metric == "min_price":
                pick = "min"
            elif metric == "max_price":
                pick = "max"
            else:
                pick = "min" if _MIN_WORD.search(low) else "max"
        # the group field itself must not also be fixed to a single value by a filter ("fuel" + "diesel"...)
        return Route(INVENTORY, {**base, "mode": "group", "field": group, "metric": metric, "pick": pick}, "inventory group-by")
    if _AVG_WORD.search(low) and re.search(r"\b(?:price|cost|mileage|km|kms)\b", low):
        return Route(INVENTORY, {**base, "mode": "average", "metric": _agg_metric(low)}, "inventory average")
    if re.search(r"\bprice range\b|\brange of (?:the )?prices?\b|\bmin(?:imum)? and max(?:imum)? price\b|\bprices? range\b", low):
        return Route(INVENTORY, {**base, "mode": "range"}, "inventory price range")
    return None


_COMPARE_WORD = re.compile(r"\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?(?=\s|$)|\bdifferences?\b")
# a verdict ("which is better / cheaper", "should I buy") is a judgement call, not a data lookup
_COMPARE_VERDICT = re.compile(r"\bwhich\b|\bbetter\b|\bbest\b|\bworth\b|\brecommend|\bshould i\b|\bcheaper\b|\bcostlier\b|\bsuggest")
_NEW_OR_USED = re.compile(r"\b(?:new|used|pre[\s-]?owned|second[\s-]?hand)\b")


def compare_route(low: str) -> Route | None:
    """'compare Swift and Baleno' / 'Creta vs Seltos' -> side-by-side from stock. Verdict
    questions, new/used qualifiers and fewer than two distinct models stay on the LLM path."""
    if not _COMPARE_WORD.search(low) or _COMPARE_VERDICT.search(low) or _NEW_OR_USED.search(low):
        return None
    models = list(dict.fromkeys(inventory.catalog().find_models(low)))
    if len(models) < 2 or len(models) > 4:
        return None
    f = {"models": [mo for _, mo in models], "makes": list(dict.fromkeys(mk for mk, _ in models))}
    return Route(INVENTORY, {"filters": f, "mode": "compare", "compare": models}, "inventory compare")


_DETAIL_CUE = re.compile(r"\b(details?|about|info|information|tell|show|price|cost|how much|available|in stock|describe|specs?)\b")


def detail_route(low: str) -> Route | None:
    m = _STOCK_REF.search(low)
    if not m or _PERSONAL.search(low) or _BOOK_VERB.search(low) or _CANCEL_VERB.search(low):
        return None
    if not (_DETAIL_CUE.search(low) or len(low.split()) <= 3):
        return None
    sid = "S-" + re.sub(r"\D", "", m.group(0))
    return Route(INVENTORY, {"filters": {}, "mode": "detail", "stock_id": sid, "sticky": False}, "inventory detail")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def context_variant_route(low: str, ctx: RouteContext) -> Route | None:
    """'more details about Sport line' right after a BMW list: a variant name alone is
    ambiguous across the catalogue, but not among the vehicles just shown. Match it against
    the vehicles in context (shown results, else the active filters) and answer about them."""
    if (_BOOK_VERB.search(low) or _CANCEL_VERB.search(low) or is_select_vehicle(low) or _COMPARE.search(low)
            or _INVENTORY_RESET.search(low)):
        return None
    ids = _context_variant_ids(low, ctx)
    if not ids:
        return None
    if len(ids) == 1:
        return Route(INVENTORY, {"filters": {}, "mode": "detail", "stock_id": ids[0], "sticky": False}, "variant in context")
    f = {**(ctx.last_filters or {}), "stock_ids": ids}
    return Route(INVENTORY, {"filters": f, "mode": "list"}, "variants in context")


def _context_variant_ids(low: str, ctx: RouteContext) -> list[str]:
    """Stock ids among the vehicles in context whose variant name the message mentions."""
    if not (ctx.last_result_ids or ctx.last_filters):
        return []
    cat = inventory.catalog()
    if cat.find_models(low):                        # the message names its own model - normal path handles it
        return []
    if ctx.last_result_ids:
        scope = [r for sid in ctx.last_result_ids if (r := cat.get(sid))]
    else:
        lf = ctx.last_filters or {}
        if not (lf.get("makes") or lf.get("models")):
            return []
        scope = cat.search(makes=lf.get("makes"), models=lf.get("models"), limit=50)[1]
    models = list(dict.fromkeys(r["model"] for r in scope))
    found = {(mo, v) for _, mo, v in cat.find_variants(low, models)} if models else set()
    return [r["stock_id"] for r in scope if (r["model"], r.get("variant")) in found]


_HANDOFF_Q = re.compile(r"\b(?:financ\w*|loans?|emis?|exchange|trade[\s-]?in)\b")

ROUTER_MAX_CHARS = 1000


def route(question: str, ctx: RouteContext) -> Route:
    text = (question or "").strip()
    low = normalize.normalize(_low(text))
    log = __import__("logging").getLogger("router")
    log.info("route request: customer_bound=%s customer_id=%s question=%r", ctx.customer_bound, ctx.customer_id, text)
    if not low:
        log.warning("empty route input: customer_id=%s", ctx.customer_id)
        return Route(CLARIFY, {"kind": "empty"}, "empty")
    if len(low) > ROUTER_MAX_CHARS:
        # The rules are tuned for chat-sized messages; very long input only costs CPU on
        # the event loop here and gets no better answer from them.
        log.info("router skipped: input too long (%d chars)", len(low))
        return Route(LLM, {}, "input too long for rules", needs_appts=True)

    if ctx.pending and ctx.pending.get("type") == "UPDATE_NUMBER":
        num = ctx.pending.get("number")
        short = len(low.split()) <= 6
        if short and re.search(r"\b(alt|alternate|alternative|secondary)\b", low):
            return Route(UPDATE, {"alt_mobile_no": num, "problems": [], "refused": []}, "slot-fill: alternate")
        if short and re.search(r"\b(primary|main|registered|first|mobile|phone|sign ?in|login)\b", low):
            return Route(UPDATE, {"mobile_no": num, "problems": [], "refused": []}, "slot-fill: primary")
    if ctx.pending and ctx.pending.get("type") == "BOOK_APPOINTMENT":
        appt_when = parse_when(text, clock.today())
        # Merge previously collected date/time from pending into the freshly
        # parsed When so that multi-turn slot-fill doesn't "forget" earlier values.
        # E.g. user gave date on turn 1, time was rejected on turn 2, user re-sends
        # a valid time on turn 3 -> the date from turn 1 must survive.
        if appt_when.date is None and not appt_when.date_error:
            saved_date = ctx.pending.get("date")
            if saved_date:
                try:
                    from datetime import date as _date
                    appt_when.date = _date.fromisoformat(saved_date)
                except (ValueError, TypeError):
                    pass
        if appt_when.hour is None and not appt_when.time_error:
            saved_hour = ctx.pending.get("hour")
            if saved_hour is not None:
                appt_when.hour = int(saved_hour)
                appt_when.minute = int(ctx.pending.get("minute") or 0)
        if appt_when.ampm is None:
            appt_when.ampm = ctx.pending.get("ampm")      # keep answering in the clock style they used earlier
        if appt_when.date is not None or appt_when.hour is not None or appt_when.date_error or appt_when.time_error or appt_when.date_vague or appt_when.time_vague:
            return Route(BOOK, {
                "appointment_type": ctx.pending.get("appointment_type"),
                "when": appt_when,
                "notes": ctx.pending.get("notes"),
            }, "slot-fill: appointment")
    if ctx.has_pending and _YES.match(low):
        return Route(CONFIRM, {"answer": "yes"}, "confirm")
    if ctx.has_pending and _NO.match(low):
        return Route(CONFIRM, {"answer": "no"}, "confirm")
    if _ACK.match(low):
        return Route(GREETING, {}, "acknowledgement")
    if _YES.match(low) or _NO.match(low):
        return Route(CLARIFY, {"kind": "nothing_pending"}, "no pending action")
    if _ABOUT.match(low):
        return Route(GREETING, {}, "about")
    if _HUMAN.search(low) and not (_BOOK_VERB.search(low) or _CANCEL_VERB.search(low)):
        return Route(CLARIFY, {"kind": "human"}, "human handoff request")
    if _GREET.match(low):
        return Route(GREETING, {}, "greeting")
    if _COMPLIMENT.match(low) or _COMPLIMENT_EXPECTATION.match(low):
        return Route(GREETING, {"kind": "compliment"}, "compliment")
    if _looks_gibberish(low):
        return Route(CLARIFY, {"kind": "gibberish"}, "low-information input")
    if re.search(r"\bbest\s+(vehicle|car|suv)\b", low) and re.search(r"\b(inventory|stock|available|have)\b", low):
        return Route(CLARIFY, {"kind": "inventory_best"}, "best vehicle needs criteria")

    if re.search(r"\b(my |the )?(last|previous|earlier) (query|question|message|request)\b|what did i (just )?(ask|say)", low):
        return Route(HISTORY, {}, "session history")

    sec = _security(text, low, ctx)
    if sec:
        return sec
    if (_MODEL_LIST_REQUEST.search(low) or _INVENTORY_FOLLOWUP.search(low)) and ctx.previous_question:
        prev = normalize.normalize(_low(ctx.previous_question))      # is_inventory() expects lower-case text
        follow_up = f"{prev} {low}"
        if is_inventory(prev) or is_inventory(follow_up) or _INVENTORY_CONTEXT.search(prev):
            result = _h_inventory(follow_up, follow_up, ctx.last_filters, ctx.last_result_ids)
            if _INVENTORY_FOLLOWUP.search(low):
                result.slots["mode"] = "models"
            return result
    # "exchange offers on the Creta?" / "Nexon on EMI?" - a finance/exchange question even when it
    # names a car; we hold no such data, so it gets the team-callback offer, not a stock listing.
    hm = _HANDOFF_Q.search(low)
    if hm and not (_BOOK_VERB.search(low) or _CANCEL_VERB.search(low)):
        if not ctx.customer_bound:
            return Route(NEED_CUSTOMER, {}, "guest")
        return Route(NOT_IN_RECORD, {"topic": hm.group(0)}, "finance/exchange handoff")
    cv = context_variant_route(low, ctx)
    if cv is not None:
        return cv
    # Conversational continuation with active filters: "other options", "more",
    # "anything else", "do you have other" etc. — reuse last inventory filters
    # so the customer stays in their browsing context instead of hitting the LLM.
    if _INVENTORY_CONTINUATION.search(low) and ctx.last_filters:
        return _h_inventory(text, low, ctx.last_filters, ctx.last_result_ids)
    if is_select_vehicle(low):
        stock = _STOCK_REF.search(low)
        return Route(SELECT_VEHICLE, {"stock_id": stock.group(0).upper()}, "vehicle selection")
    dr = detail_route(low)
    if dr is not None:
        return dr
    if _VEHICLE_DETAILS.search(low):
        makes = inventory.catalog().find_makes(low)
        if makes and not inventory.catalog().find_models(low):
            model_match = re.search(r"\b(?:" + "|".join(re.escape(make.lower()) for make in makes) + r")\s+([a-z0-9][a-z0-9 -]{1,30})", low)
            if model_match and model_match.group(1).split()[0] not in _GENERIC_AFTER_MAKE and not model_match.group(1)[0].isdigit():
                model = model_match.group(1).strip(" .,-")
                return Route(CLARIFY, {"kind": "vehicle_unavailable", "make": makes[0], "model": model}, "vehicle not in inventory")

    cache_route: dict[str, Route | None] = {}
    unmatched: list[str] = []
    fams = _families(low, cache_route, unmatched)
    if fams and unmatched:
        # "what's my email and does the Creta have a sunroof?" - answering only the email
        # would silently drop the second question; let the LLM see the whole message.
        log.info("router partial match -> LLM: families=%s unmatched=%r", fams, unmatched)
        return Route(LLM, {}, "multi-clause: unmatched clause", needs_appts=True,
                     hints={"multi": True, "families": fams, "unmatched": unmatched})
    if not fams and is_update(low):
        fams.append("update")
    if not fams:
        r0 = record_route(low)
        if r0 is not None:
            if r0.intent != LLM:
                fams.append("record")
                cache_route["record"] = r0
            else:
                # record_route() itself decided this needs the full LLM (e.g. a contact
                # window needing date reasoning) - return its specific reason directly
                # rather than losing it to the generic "no rule matched" fallback below,
                # which the general planner handles directly.
                r0.needs_appts = True
                return r0
    # Anything not caught above (including inventory/record phrasing the strict regexes
    # above don't recognise) falls through to "no rule matched" and gets a shot at the
    # small-LLM intent stage in service.py, instead of a second,
    # independently-tuned fuzzy guesser here.
    if "inventory" not in fams and "register" not in fams and (is_inventory(low) or (ctx.last_filters and _is_filter_only_refinement(low))):
        fams.append("inventory")
    if "register" in fams:                       # registration text carries name / vehicle / mobile / email data
        fams = [f for f in fams if f not in ("inventory", "record", "update")]

    if not fams or fams == ["inventory"] or (fams == ["record"] and not re.search(r"\b(my|mine|i|me)\b", low)):
        cr = compare_route(low)
        if cr is not None:
            return cr
        ar = aggregate_route(text, low, ctx.last_result_ids)
        if ar is not None:
            return ar

    if len(fams) >= 2:
        hints: dict[str, Any] = {"multi": True, "families": fams}
        return Route(LLM, {}, "multi-intent: " + "+".join(fams), needs_appts=True, hints=hints)
    if not fams:
        # ---- check if the query should be routed to SQL (analytical/aggregate) ------
        # This runs BEFORE the LLM fallback so that SQL-suitable queries get fast answers.
        from .. import sql_engine
        if sql_engine.ready() and sql_engine.should_route_to_sql(low):
            log.info("router -> SQL_QUERY: question=%r", text)
            return Route(SQL_QUERY, {"question": text}, "sql analytical query")
        log.warning("router fallback to LLM: customer_id=%s question=%r", ctx.customer_id, text)
        return Route(LLM, {}, "no rule matched", needs_appts=True)

    fam = fams[0]

    # Even within inventory family, analytical queries (avg price, group by, etc.)
    # should route to SQL for better results than the simple filter-based search.
    if fam == "inventory":
        from .. import sql_engine
        if sql_engine.ready() and sql_engine.should_route_to_sql(low):
            return Route(SQL_QUERY, {"question": text}, "sql analytical query (inventory)")

    if fam == "oos":
        return Route(REFUSE, {"kind": "off_topic"}, "off topic")
    if fam == "reschedule":
        return Route(CLARIFY, {"kind": "reschedule"}, "reschedule")
    if fam == "register":
        return _h_register(text, low, ctx)
    if fam == "inventory":
        # "which ones are automatic?" / "which one is cheapest?" carry a concrete filter or ranking, so the
        # inventory engine answers them exactly (against the previously shown set when there is one).
        # "which car should I buy?" has neither - that is advice, and stays with the LLM.
        concrete = bool(X.inventory_filters(text))
        if _COMPARE.search(low) or _ADVICE.search(low) or (_WHICH_PICK.search(low) and not concrete):
            return Route(LLM, {}, "inventory comparison / advice", hints={"inventory": True})
        return _h_inventory(text, low, ctx.last_filters, ctx.last_result_ids)
    if fam == "select_vehicle":
        stock = _STOCK_REF.search(low)
        return Route(SELECT_VEHICLE, {"stock_id": stock.group(0).upper()}, "vehicle selection")

    # everything below needs a bound customer
    if not ctx.customer_bound:
        return Route(NEED_CUSTOMER, {}, "guest")
    if fam == "book":
        r = _h_book(text, low)
        if r.intent == BOOK and not r.slots.get("notes"):
            # "book a test drive for the sport line" right after seeing three BMWs
            ids = _context_variant_ids(low, ctx)
            if len(ids) == 1 and (row := inventory.catalog().get(ids[0])):
                r.slots["notes"] = f"{row['make']} {row['model']} {row['variant']} ({row['stock_id']})"
        return r
    if fam == "cancel":
        return _h_cancel(text, low)
    if fam == "update":
        return _h_update(text, low)
    r = record_route(low) or cache_route.get("record")
    if r is None:
        return Route(LLM, {}, "record intent not resolved", needs_appts=True)
    if r.intent == LLM:
        r.needs_appts = True
    return r