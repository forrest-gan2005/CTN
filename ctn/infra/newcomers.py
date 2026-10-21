"""Citizens who registered after the start. The organisers keep adding them all competition (about 6000 in all, 2343
before the start), so the whitelist of accounts there before the start misses most of them. They are told apart from
competitors' accounts by the names the citizen generator gives them and by how little they post
(data/research/post-start-npcs-20261021.md: ~99% precision on tier A, ~94% on tier C that posted)."""
import re
from collections import Counter, defaultdict, namedtuple
from typing import Iterable, Optional

LAST_PRE_START_ID = 2343  # the newest account there before the start; those are whitelisted already
# Behaviour is judged over the latest WINDOW_HOURS only: the limits were measured over two days, and a count since the
# start would leave out every citizen as the days pass.
WINDOW_HOURS = 48
MAX_HOURS = 4  # distinct UTC hours with a post; citizens' p99 was 3, known bots' median 16
MAX_POSTS = 30  # citizens' most was 24, known bots' median 40
COPY_CHARS = 40  # a longer text another account posted word for word: citizens never reuse text

FULL_NAME = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z']*)*$")  # 99.3% of citizens, 60% of known bots
VERBIN = r"(?:[a-z]{2,8}in)"  # chillin, sprintin, bouncin, wanderin, rantin ...
ANIMALS = ("bear|cat|dog|gator|heron|lynx|moose|bunny|tiger|raven|parrot|pelican|lion|dolphin|octopus|python|gull|"
           "hedgehog|jaguar|falcon|camel|badger|goat|oat|owl|fox|wolf|otter|shark|crab|whale|eagle|hawk|swan|frog|"
           "duck|turtle|panda|koala|zebra|llama|seal|walrus|bobcat|cougar|wombat|possum")
SUFFIXES = ("rules|boost|live|4eva|era|again|returns|ftw|fast|prime|rocks|wave|irl|official|zone|plus|hq|daily|vibes|"
            "life|now|pro|fan|here|time")
# Families whose shape alone gives the generator away: a word outside the vocabulary doesn't count against them.
STRONG = {"initials_camel", "initials_lower", "initials_middle", "i_have_N_things", "post_about_X_Y", "first_is_so_X",
          "verbin_with_first", "Verbin_First_NN", "TheReal/IAm", "First__NN", "VerbinAnimal", "AdjNounNNN",
          "First_Animalv"}
# For an account we know only by its username: (family, pattern, whether its words must be generator words).
USERNAME_ONLY = [
    ("initials_camel", r"(?:^|[a-z_\d])[A-Z][a-z][A-Z][a-z](?:[A-Z_\d]|$)", False),
    ("xx_xx", r"(?i)^xx_[a-z]+_xx$", False),
    ("i_have", r"^i_have_\d+_[a-z]+$", False),
    ("post_about", r"^post_about_[a-z]+_[a-z]+$", False),
    ("verbin_with", r"(?i)^[a-z]{2,8}in_with_[a-z]+$", False),
    ("is_so", r"(?i)^[a-z]+_is_so_[a-z]+$", False),
    ("TheReal/IAm", r"^(TheReal|IAm)[A-Z][a-z]+[A-Z][a-z]+$", False),
    ("Animalv", r"^[A-Za-z]+_[A-Z][a-z]+v\d$", False),
    ("First__NN", r"^[A-Za-z]+__\d+$", False),
    ("Verbin_X_NN", r"(?i)^[a-z]{2,8}in_[a-z]+_\d+$", False),
    ("VerbinAnimal", r"^[A-Z][a-z]+in[A-Z][a-z]+$", False),
    ("AdjNounNNN", r"^[A-Z][a-z]+[A-Z][a-z]+\d{3}$", False),
    ("the_adj_noun", r"^the_(?!real_)[a-z]+_[a-z]+$", True),
    ("job_Suffix", r"^[a-z]+_[a-z]+_(Rules|Boost|live|4eva|Era|Again|Returns|FTW|Fast|Prime|Rocks|Wave|Here|Time)$",
     True),
    ("not_adj_first", r"^not_[a-z]+_[a-z]+$", None),  # its adjective must be a generator word
    ("prefix_first", r"^(only|just|actually|literally|im|ask|its|call_me|real)_[a-z]+$", True),
    ("first_the_noun", r"^[a-z]+_the_[a-z]+$", True),
    ("adj_fila_suffix", r"^[a-z]+_[a-z]{4}_(rules|boost|live|4eva|era|again|returns|ftw|fast|prime|rocks|wave|here|"
                        r"time|zone|now)$", False),
]


def _template(pattern: str):
    """A username template compiled once: matched against "first|last|l|username", so {f}, {l} and {l0} (the display
    name's first name, last name and last initial) are backreferences instead of a pattern per account."""
    slots = pattern.replace("{f}", "(?P=f)").replace("{l}", "(?P=l)").replace("{l0}", "(?P=l0)")
    return re.compile(r"(?P<f>[a-z]+)\|(?P<l>[a-z]*)\|(?P<l0>[a-z]?)\|" + slots)


# Name templates in order: "A" the generator's, "C" plain names (teams use those too).
TEMPLATES = [(tier, name, _template(pattern)) for tier, name, pattern in [
    ("A", "first_is_so_X", r"{f}_is_so_[a-z]+"),
    ("A", "verbin_with_first", VERBIN + r"_with_{f}"),
    ("A", "Verbin_First_NN", VERBIN + r"_{f}_\d+"),
    ("A", "First_NN_Animal", r"{f}\d+[a-z]+"),
    ("A", "TheReal/IAm", r"(thereal|iam){f}{l}"),
    ("A", "First__NN", r"{f}__\d+"),
    ("A", "prefix_first", r"(only|just|actually|literally|im|ask|real|call_me|the|not)_{f}"),
    ("A", "not_adj_first", r"not_[a-z]+_{f}"),
    ("A", "first_the_noun", r"{f}_the_[a-z]+"),
    ("A", "first_job_Suffix", r"{f}{l0}?_[a-z]+_(" + SUFFIXES + ")"),
    ("A", "W_W_first", r"(?!the_real_)[a-z]+_[a-z]+_{f}"),
    ("A", "WordFirstWord", r"[a-z]{3,}{f}[a-z]{3,}"),
    ("A", "First_Animalv", r"{f}_[a-z]+v\d"),
    ("A", "first_adj_NN", r"{f}_[a-z]+_\d+"),
    ("A", "its_first", r"its_{f}"),
    ("A", "Adj_First_X", r"[a-z]+_{f}_(official|irl|hq|online|zone)"),
    ("A", "firstl_job", r"{f}{l0}_[a-z]+"),
    ("A", "wordfirstplus", r"[a-z]+{f}(plus|prime|max|pro|irl|hq|live)"),
    ("A", "W_first_W", r"[a-z]+_{f}_[a-z]+"),
    ("C", "firstlYYYY", r"{f}{l0}(19[3-9]\d|200\d)"),
    ("C", "first.last", r"{f}[._]?{l}\d*"),
    ("C", "firstNN", r"{f}_?\d+"),
    ("C", "wordfirst", r"[a-z]+{f}\d*"),
]]

# The generator's slot words (scholar, swimmer, silver, falcon, ...) and citizens' first names, learnt from the
# accounts there before the start.
Vocab = namedtuple("Vocab", "words firsts")


def _names(display_name: Optional[str]) -> Optional[tuple]:
    """(first, last, words) of a display name, lower case, or None without two words."""
    words = re.sub(r"[^A-Za-z ]", "", display_name or "").split()
    if len(words) < 2 or len(words[0]) < 2:
        return None
    return words[0].lower(), words[-1].lower(), words


def _tokens(username: str) -> list:
    out = []
    for part in re.split(r"[_.\d]+", username):
        out += [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", part)]
    return [t for t in out if t]


def _name_parts(display_name: Optional[str]) -> set:
    found = _names(display_name)
    if not found:
        return set()
    f, l, words = found
    parts = {f, l, f[:2], l[:2], f[:2] + l[:2], f[:1], l[:1]}
    for w in words:
        parts |= {w.lower(), w.lower()[:2]}
    return parts


def family(username: str, display_name: Optional[str]) -> Optional[tuple]:
    """("A", family) for a generator template, ("C", family) for a plain-name form, or None. The display name fills
    the name slots; no check here that its words are the generator's."""
    u, ul = username, username.lower()
    if re.fullmatch(r"i_have_\d+_[a-z]+", u):
        return "A", "i_have_N_things"
    if re.fullmatch(r"post_about_[a-z]+_[a-z]+", u):
        return "A", "post_about_X_Y"
    if re.fullmatch(r"the_(?!real_)[a-z]+_[a-z]+", u):
        return "A", "the_adj_noun"
    found = _names(display_name)
    if (re.fullmatch(r"[A-Z][a-z]+[A-Z][a-z]+\d{3}", u)
            or re.fullmatch(r"[a-z]{6,}\d{3}", u) and not (found and (found[0] in ul or found[1] in ul))):
        return "A", "AdjNounNNN"
    if re.fullmatch(r"[A-Z][a-z]+in[A-Z][a-z]+", u) or re.fullmatch(rf"[a-z]+in(?:{ANIMALS})", u):
        return "A", "VerbinAnimal"
    if not found:
        return None
    f, l, words = found
    token = f[:2] + l[:2]
    if len(l) >= 2 and token[0].upper() + token[1] + token[2].upper() + token[3] in u:
        return "A", "initials_camel"
    rest = ul.replace(f, "", 1)
    if (u == ul and len(l) >= 2 and token in re.sub(r"[^a-z]", "", rest)
            and not (l.startswith(token[2:]) and l in rest and token not in rest.replace(l, "", 1))):
        return "A", "initials_lower"
    if len(words) >= 3 and (f[:2] + words[-2][0]).lower() in ul and f[:2].capitalize() + words[-2][0].upper() in u:
        return "A", "initials_middle"
    subject = f"{f}|{l}|{l[:1]}|{ul}"
    for tier, name, pattern in TEMPLATES:
        if pattern.fullmatch(subject):
            return tier, name
    return None


def vocabulary(citizens: Iterable[tuple]) -> Vocab:
    """The generator's words from (username, display name) of citizens there before the start: every username word
    that isn't part of the account's own name, from the accounts whose name fits a template."""
    words, firsts = set(), set()
    for username, display_name in citizens:
        if not FULL_NAME.match(display_name or ""):
            continue
        found = _names(display_name)
        if found:
            firsts.add(found[0])
        if (family(username, display_name) or ("",))[0] != "A":
            continue
        own = _name_parts(display_name)
        words |= {t for t in _tokens(username) if t not in own and len(t) >= 2}
    return Vocab(frozenset(words), frozenset(firsts))


def _segments(text: str, words, longest: int = 14) -> bool:
    """Whether `text` cuts entirely into `words`."""
    ok = [True] + [False] * len(text)
    for i in range(1, len(text) + 1):
        ok[i] = any(ok[j] and text[j:i] in words for j in range(max(0, i - longest), i))
    return ok[-1]


def name_tier(username: str, display_name: Optional[str], vocab: Vocab) -> Optional[tuple]:
    """(tier, family) for an account whose display name we know: "A" a generator template, "C" a plain name (teams
    use those too), None neither. A template with a word the generator never used is left out unless its shape is
    one of the STRONG ones."""
    if not FULL_NAME.match(display_name or ""):
        return None
    found = family(username, display_name)
    if found is None or found[0] == "C":
        return found
    words = set(vocab.words) | _name_parts(display_name)
    if all(_segments(p, words) for p in re.split(r"[^a-z]+", username.lower()) if p) or found[1] in STRONG:
        return found
    return None


def username_tier(username: str, vocab: Vocab) -> Optional[tuple]:
    """("U", family) for an account we know only by its username (missing from the account watcher's list): the
    template shapes, with the name slots checked against the generator's words and citizens' first names. It finds
    about 59% of citizens."""
    for name, pattern, checked in USERNAME_ONLY:
        if not re.search(pattern, username):
            continue
        words = [w for w in _tokens(username) if len(w) > 2]
        if checked and not all(w in vocab.words or w in vocab.firsts for w in words):
            return None
        if checked is None and (len(words) < 2 or words[1] not in vocab.words):
            return None
        return "U", name
    return None


def activity(rows: Iterable[tuple]) -> dict:
    """key -> (posts, distinct UTC hours, whether one was a copy) from (key, text, created_at UTC ISO) of the posts in
    the window; key is the account id. A copy: a text over COPY_CHARS another key posted word for word."""
    posts, hours, by_text = Counter(), defaultdict(set), defaultdict(set)
    for key, content, created_at in rows:
        posts[key] += 1
        if created_at:
            hours[key].add(created_at[:13])
        text = (content or "").strip().lower()
        if len(text) > COPY_CHARS:
            by_text[text].add(key)
    copied = {k for keys in by_text.values() if len(keys) > 1 for k in keys}
    return {k: (n, len(hours[k]), k in copied) for k, n in posts.items()}


def behaves(active: Optional[tuple]) -> bool:
    """Posts like a citizen over the window, given its activity() row (None: no post in the window)."""
    if active is None:
        return True
    posts, hours, copied = active
    return hours <= MAX_HOURS and posts <= MAX_POSTS and not copied


def label(accounts: dict, posted: set, active: dict, excluded: set, vocab: Vocab) -> dict:
    """account id -> tier of every account taken for a citizen who registered after the start:
    "A" a template name that posts like a citizen, "A-silent" one never seen posting (on its name alone), "C" a plain
    name that has posted and posts like a citizen (never posted, only ~60% are citizens: left out), "U" a template
    username of an account whose display name we don't have.
    accounts: id -> (username, display name, listed): listed False when only its posts show it (no display name
    known). posted: ids with any stored post. active: activity() over the window. excluded: ids never labelled
    (ours, the organisers' scripted fleet, known competitors, renamed accounts)."""
    out = {}
    for key, (username, display_name, listed) in accounts.items():
        if key is None or key <= LAST_PRE_START_ID or key in excluded:
            continue
        found = name_tier(username, display_name, vocab) if listed else username_tier(username, vocab)
        if found is None or not behaves(active.get(key)):
            continue
        tier = found[0]
        if key not in posted:
            if tier != "A":
                continue
            tier = "A-silent"
        out[key] = tier
    return out


def newcomers_check() -> None:
    """Generator names pass and the rival swarms seen on 20-21 October fail: a citizen-style name posting 40 times a
    day, plain-name copies posted word for word, name_job and numbered series."""
    pre = [("GolferRuMaViolet", "Ruth Mason"), ("Mary_is_so_emerald", "Mary Hale"), ("chillin_with_linh", "Linh Tran"),
           ("i_have_15_pythons", "Owen Park"), ("post_about_kraken_otters", "Ada Bloom"),
           ("the_mighty_teapot", "Rita Moss"), ("HungryBeacon895", "Gus Lane"), ("derek_the_phoenix", "Derek Ross"),
           ("jacob_birder_Rules", "Jacob Hill"), ("susan.perkins", "Susan Perkins"),
           ("your_pal_maria", "Maria Lopez"), ("swift_meadow4519", "Taylor B."), ("old_salt", None)]
    vocab = vocabulary(pre)
    assert {"golfer", "violet", "emerald", "chillin", "pythons", "kraken", "otters", "mighty", "teapot", "hungry",
            "beacon", "phoenix", "birder", "rules"} <= vocab.words, sorted(vocab.words)
    assert not {"ruth", "ru", "ma", "perkins", "swift", "meadow"} & vocab.words, "names and non-templates aren't words"
    assert {"ruth", "mary", "susan"} <= vocab.firsts and "taylor" not in vocab.firsts

    good = {"BoSa37Falcon": "Bob Sands", "ReginaReHo241": "Regina Holt", "Foggy_Ferry_EvFr": "Eve Frost",
            "xX_PaulPaBu_Xx": "Paul Burke", "Nina_is_so_violet": "Nina Cole", "sprintin_with_omar": "Omar Haddad",
            "TheRealDonnieHuber": "Donnie Huber", "IAmRuthHerman": "Ruth Herman", "i_have_3_otters": "Kai West",
            "post_about_teapot_kraken": "Lou Grant", "the_hungry_phoenix": "Ian Moss", "Sandra__55": "Sandra Pyle",
            "Kathi_Dolphinv7": "Kathi Brand", "MightyBeacon204": "Jo Rees", "derek_the_teapot": "Derek Stone"}
    for username, name in good.items():
        assert name_tier(username, name, vocab) and name_tier(username, name, vocab)[0] == "A", username
    plain = {"david_fletcher": "David Fletcher", "adriennem1992": "Adrienne Moss", "regina_84": "Regina Lowe"}
    assert all(name_tier(u, n, vocab) and name_tier(u, n, vocab)[0] == "C" for u, n in plain.items())
    assert name_tier("Nina_is_so_violet", "nina", vocab) is None, "a display name that isn't First Last"
    assert name_tier("Nina_is_so_violet", None, vocab) is None
    assert name_tier("Nina_is_so_violet", "🌊 Nina", vocab) is None
    rivals = {"claire_bookshop_newhaven": "Claire Dumont", "hollie_studentnurse": "Hollie Jones", "ducthang_004": None,
              "civic_forum_12_9a87aa": "Civic Forum 12", "Swan_150_007": None,
              "ctnyes_policy_015c18": "Kingstonia Policy Notes", "derek_the_zeppelin": "Derek Stone"}
    for username, name in rivals.items():
        assert name_tier(username, name, vocab) is None, username  # zeppelin: never a generator word
    assert name_tier("your_pal_dennis", "Dennis Kalani", vocab), "a rival's generator name: behaviour must catch it"
    assert username_tier("ArAg_Cranky_12", vocab)[0] == "U" and username_tier("i_have_9_otters", vocab)
    assert username_tier("the_mighty_teapot", vocab) and not username_tier("the_coastal_zeppelin", vocab)
    assert not username_tier("claire_bookshop_newhaven", vocab) and not username_tier("civic_forum_12_9a87aa", vocab)

    # 1 a citizen, 2 a citizen never seen posting, 3 a rival with a generator name posting all day, 4/5 a swarm
    # posting one text, 6 a plain-name account never seen posting, 7 a known competitor, 8 there before the start,
    # 9 a template username without a display name, 10 an account on the list without a display name.
    accounts = {2401: ("Mary_is_so_violet", "Mary Lane", True), 2402: ("BoSa37Falcon", "Bob Sands", True),
                3400: ("your_pal_dennis", "Dennis Kalani", True), 5001: ("anna_smith_21", "Anna Smith", True),
                5002: ("ben_jones_22", "Ben Jones", True), 5003: ("carl_white", "Carl White", True),
                5004: ("TheRealDonnieHuber", "Donnie Huber", True), 1200: ("GolferRuMaViolet", "Ruth Mason", True),
                9000: ("i_have_9_otters", None, False), 9001: ("Nina_is_so_violet", None, True)}
    rows = [(2401, "the ferry was late again", "2026-10-21T08:10:00+00:00"),
            (2401, "anyone else at the market", "2026-10-21T09:10:00+00:00"),
            (9000, "lovely morning on the harbour", "2026-10-21T09:00:00+00:00")]
    rows += [(3400, f"vote no, post {i}", f"2026-10-21T{i % 24:02d}:{i % 60:02d}:00+00:00") for i in range(40)]
    swarm = "I kept flip-flopping on this, but a simple check beats another winter of closure."
    rows += [(5001, swarm, "2026-10-21T01:24:00+00:00"), (5002, swarm, "2026-10-21T01:27:00+00:00")]
    rows += [(1200, "pre-start citizen", "2026-10-21T10:00:00+00:00")]
    active = activity(rows)
    assert active[2401] == (2, 2, False) and active[3400][1] > MAX_HOURS and active[5001][2] and active[5002][2]
    posted = {2401, 3400, 5001, 5002, 9000, 1200, 5004}
    got = label(accounts, posted, active, excluded={5004}, vocab=vocab)
    assert got == {2401: "A", 2402: "A-silent", 9000: "U"}, got
    many = [(2401, f"post {i}", f"2026-10-21T0{i % 3}:00:00+00:00") for i in range(MAX_POSTS + 1)]
    assert not label({2401: accounts[2401]}, {2401}, activity(many), set(), vocab), "over MAX_POSTS"
    print("newcomers ok")


if __name__ == "__main__":
    newcomers_check()
