"""Offline reference tables.

Deliberately hand-maintained rather than fetched, so the tool works fully
air-gapped. These encode the "cheap" geolocation cues that experienced
analysts check first: which side of the road traffic drives on, which
scripts and languages are used where, and which top-level domains and
phone prefixes map to which country.

All country codes are ISO-3166-1 alpha-2.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Driving side. One of the highest-value single cues in a street-level photo:
# it splits the world roughly 35/165 by country and is usually unambiguous.
# ---------------------------------------------------------------------------
LEFT_DRIVING: frozenset[str] = frozenset({
    "AG", "AI", "AU", "BB", "BD", "BM", "BN", "BS", "BT", "BW", "CC", "CK",
    "CY", "CX", "DM", "FJ", "FK", "GB", "GD", "GG", "GY", "HK", "ID", "IE",
    "IM", "IN", "JE", "JM", "JP", "KE", "KI", "KN", "KY", "LC", "LK", "LS",
    "MO", "MS", "MT", "MU", "MV", "MW", "MY", "MZ", "NA", "NF", "NP", "NR",
    "NU", "NZ", "PG", "PK", "PN", "SB", "SC", "SG", "SH", "SR", "SZ", "TC",
    "TH", "TL", "TO", "TT", "TV", "TZ", "UG", "VC", "VG", "VI", "WS", "ZA",
    "ZM", "ZW",
})

# ---------------------------------------------------------------------------
# Writing systems -> countries where that script is in routine public use.
# Used to convert an OCR script detection into a country prior.
# ---------------------------------------------------------------------------
SCRIPT_COUNTRIES: dict[str, list[str]] = {
    "Latin": [],  # too broad to constrain; handled as a no-op
    "Cyrillic": ["RU", "UA", "BY", "BG", "RS", "MK", "KZ", "KG", "TJ", "MN", "ME", "UZ"],
    "Greek": ["GR", "CY"],
    "Arabic": ["SA", "EG", "AE", "IQ", "JO", "SY", "LB", "KW", "QA", "BH", "OM",
                "YE", "LY", "TN", "DZ", "MA", "SD", "PS", "IR", "AF", "PK"],
    "Hebrew": ["IL"],
    "Han": ["CN", "TW", "HK", "MO", "SG", "JP"],
    "Hiragana": ["JP"],
    "Katakana": ["JP"],
    "Hangul": ["KR", "KP"],
    "Thai": ["TH"],
    "Lao": ["LA"],
    "Khmer": ["KH"],
    "Myanmar": ["MM"],
    "Devanagari": ["IN", "NP"],
    "Bengali": ["BD", "IN"],
    "Tamil": ["IN", "LK", "SG", "MY"],
    "Telugu": ["IN"],
    "Kannada": ["IN"],
    "Malayalam": ["IN"],
    "Gujarati": ["IN"],
    "Gurmukhi": ["IN"],
    "Sinhala": ["LK"],
    "Georgian": ["GE"],
    "Armenian": ["AM"],
    "Ethiopic": ["ET", "ER"],
    "Tibetan": ["CN", "BT", "NP"],
    "Mongolian": ["CN", "MN"],
}

# Latin-script languages still carry signal through diacritics and stopwords.
LANGUAGE_COUNTRIES: dict[str, list[str]] = {
    "en": ["GB", "US", "AU", "NZ", "IE", "CA", "ZA", "IN", "SG", "MT", "PH"],
    "es": ["ES", "MX", "AR", "CO", "CL", "PE", "VE", "EC", "GT", "CU", "BO",
            "DO", "HN", "PY", "SV", "NI", "CR", "PA", "UY"],
    "pt": ["PT", "BR", "AO", "MZ", "CV", "GW", "ST", "TL"],
    "fr": ["FR", "BE", "CH", "CA", "LU", "MC", "SN", "CI", "ML", "BF", "NE",
            "TD", "CM", "GA", "CG", "CD", "MG", "HT", "DZ", "MA", "TN"],
    "de": ["DE", "AT", "CH", "LI", "LU", "BE"],
    "it": ["IT", "CH", "SM", "VA"],
    "nl": ["NL", "BE", "SR"],
    "pl": ["PL"],
    "cs": ["CZ"],
    "sk": ["SK"],
    "hu": ["HU"],
    "ro": ["RO", "MD"],
    "hr": ["HR", "BA"],
    "sl": ["SI"],
    "sv": ["SE", "FI"],
    "no": ["NO"],
    "da": ["DK", "GL"],
    "fi": ["FI"],
    "is": ["IS"],
    "et": ["EE"],
    "lv": ["LV"],
    "lt": ["LT"],
    "tr": ["TR", "CY"],
    "sq": ["AL", "XK", "MK"],
    "vi": ["VN"],
    "id": ["ID"],
    "ms": ["MY", "BN", "SG"],
    "tl": ["PH"],
    "sw": ["TZ", "KE", "UG", "CD"],
    "af": ["ZA", "NA"],
    "eu": ["ES"],
    "ca": ["ES", "AD"],
    "gl": ["ES"],
    "cy": ["GB"],
    "ga": ["IE"],
}

# ---------------------------------------------------------------------------
# ccTLDs visible on shopfronts, vans and advertising hoardings. A single
# ".hr" on a delivery van is often the whole ballgame.
# ---------------------------------------------------------------------------
CCTLD_COUNTRY: dict[str, str] = {
    ".ad": "AD", ".ae": "AE", ".af": "AF", ".al": "AL", ".am": "AM", ".ao": "AO",
    ".ar": "AR", ".at": "AT", ".au": "AU", ".az": "AZ", ".ba": "BA", ".bd": "BD",
    ".be": "BE", ".bg": "BG", ".bh": "BH", ".bo": "BO", ".br": "BR", ".bw": "BW",
    ".by": "BY", ".ca": "CA", ".cd": "CD", ".ch": "CH", ".ci": "CI", ".cl": "CL",
    ".cm": "CM", ".cn": "CN", ".co": "CO", ".cr": "CR", ".cu": "CU", ".cy": "CY",
    ".cz": "CZ", ".de": "DE", ".dk": "DK", ".do": "DO", ".dz": "DZ", ".ec": "EC",
    ".ee": "EE", ".eg": "EG", ".es": "ES", ".et": "ET", ".fi": "FI", ".fj": "FJ",
    ".fr": "FR", ".ge": "GE", ".gh": "GH", ".gr": "GR", ".gt": "GT", ".hk": "HK",
    ".hn": "HN", ".hr": "HR", ".hu": "HU", ".id": "ID", ".ie": "IE", ".il": "IL",
    ".in": "IN", ".iq": "IQ", ".ir": "IR", ".is": "IS", ".it": "IT", ".jm": "JM",
    ".jo": "JO", ".jp": "JP", ".ke": "KE", ".kg": "KG", ".kh": "KH", ".kr": "KR",
    ".kw": "KW", ".kz": "KZ", ".la": "LA", ".lb": "LB", ".lk": "LK", ".lt": "LT",
    ".lu": "LU", ".lv": "LV", ".ly": "LY", ".ma": "MA", ".md": "MD", ".me": "ME",
    ".mk": "MK", ".mm": "MM", ".mn": "MN", ".mt": "MT", ".mu": "MU", ".mv": "MV",
    ".mx": "MX", ".my": "MY", ".mz": "MZ", ".na": "NA", ".ng": "NG", ".ni": "NI",
    ".nl": "NL", ".no": "NO", ".np": "NP", ".nz": "NZ", ".om": "OM", ".pa": "PA",
    ".pe": "PE", ".ph": "PH", ".pk": "PK", ".pl": "PL", ".pt": "PT", ".py": "PY",
    ".qa": "QA", ".ro": "RO", ".rs": "RS", ".ru": "RU", ".rw": "RW", ".sa": "SA",
    ".se": "SE", ".sg": "SG", ".si": "SI", ".sk": "SK", ".sn": "SN", ".sv": "SV",
    ".sy": "SY", ".th": "TH", ".tn": "TN", ".tr": "TR", ".tw": "TW", ".tz": "TZ",
    ".ua": "UA", ".ug": "UG", ".uk": "GB", ".uy": "UY", ".uz": "UZ", ".ve": "VE",
    ".vn": "VN", ".za": "ZA", ".zm": "ZM", ".zw": "ZW",
}

# International dialling prefixes, longest-match first at lookup time.
PHONE_PREFIX_COUNTRY: dict[str, str] = {
    "20": "EG", "212": "MA", "213": "DZ", "216": "TN", "218": "LY", "220": "GM",
    "221": "SN", "225": "CI", "233": "GH", "234": "NG", "237": "CM", "251": "ET",
    "254": "KE", "255": "TZ", "256": "UG", "260": "ZM", "263": "ZW", "264": "NA",
    "265": "MW", "267": "BW", "27": "ZA", "30": "GR", "31": "NL", "32": "BE",
    "33": "FR", "34": "ES", "351": "PT", "352": "LU", "353": "IE", "354": "IS",
    "355": "AL", "356": "MT", "357": "CY", "358": "FI", "359": "BG", "36": "HU",
    "370": "LT", "371": "LV", "372": "EE", "373": "MD", "374": "AM", "375": "BY",
    "376": "AD", "377": "MC", "378": "SM", "380": "UA", "381": "RS", "382": "ME",
    "383": "XK", "385": "HR", "386": "SI", "387": "BA", "389": "MK", "39": "IT",
    "40": "RO", "41": "CH", "420": "CZ", "421": "SK", "43": "AT", "44": "GB",
    "45": "DK", "46": "SE", "47": "NO", "48": "PL", "49": "DE", "51": "PE",
    "52": "MX", "53": "CU", "54": "AR", "55": "BR", "56": "CL", "57": "CO",
    "58": "VE", "60": "MY", "61": "AU", "62": "ID", "63": "PH", "64": "NZ",
    "65": "SG", "66": "TH", "7": "RU", "81": "JP", "82": "KR", "84": "VN",
    "852": "HK", "853": "MO", "855": "KH", "856": "LA", "86": "CN", "880": "BD",
    "886": "TW", "90": "TR", "91": "IN", "92": "PK", "93": "AF", "94": "LK",
    "95": "MM", "960": "MV", "961": "LB", "962": "JO", "963": "SY", "964": "IQ",
    "965": "KW", "966": "SA", "967": "YE", "968": "OM", "970": "PS", "971": "AE",
    "972": "IL", "973": "BH", "974": "QA", "975": "BT", "976": "MN", "977": "NP",
    "98": "IR", "994": "AZ", "995": "GE", "996": "KG", "998": "UZ",
}

# ---------------------------------------------------------------------------
# Koppen-style biome bands, used to turn a CLIP vegetation/climate call into
# a latitude constraint. Ranges are absolute-latitude, generous by design.
# ---------------------------------------------------------------------------
BIOME_LAT_BANDS: dict[str, tuple[float, float]] = {
    "tropical rainforest": (0.0, 15.0),
    "tropical savanna": (5.0, 25.0),
    "hot desert": (12.0, 35.0),
    "mediterranean scrub": (28.0, 45.0),
    "temperate deciduous forest": (30.0, 58.0),
    "temperate grassland": (30.0, 55.0),
    "boreal conifer forest": (48.0, 70.0),
    "tundra": (60.0, 84.0),
    "alpine": (0.0, 90.0),
    "polar ice": (66.0, 90.0),
}

# CLIP zero-shot prompt banks. Each axis yields either a direct constraint or
# an interpretive note on the evidence board.
PROMPT_AXES: dict[str, list[str]] = {
    "biome": list(BIOME_LAT_BANDS.keys()),
    "setting": [
        "dense city centre", "suburban residential street", "small town high street",
        "industrial estate", "rural farmland", "open wilderness", "coastline",
        "mountain landscape", "desert", "forest interior", "riverside", "beach",
    ],
    "road_markings": [
        "white road markings", "yellow road markings",
        "white and yellow road markings", "unmarked road surface",
    ],
    "architecture": [
        "north american wood frame housing", "british terraced brick housing",
        "northern european modernist housing", "mediterranean stucco housing",
        "soviet era concrete apartment blocks", "east asian high rise housing",
        "south asian concrete housing", "latin american concrete housing",
        "sub saharan african housing", "middle eastern flat roofed housing",
        "traditional japanese housing", "alpine chalet architecture",
    ],
    "season": [
        "summer foliage", "autumn foliage", "bare winter trees",
        "snow on the ground", "spring blossom",
    ],
    "utility": [
        "wooden utility poles with overhead wires", "concrete utility poles",
        "metal lattice pylons", "no overhead wires, buried utilities",
    ],
}

# Architecture prompt -> country prior. Coarse by nature; weighted low.
ARCHITECTURE_COUNTRIES: dict[str, list[str]] = {
    "north american wood frame housing": ["US", "CA"],
    "british terraced brick housing": ["GB", "IE"],
    "northern european modernist housing": ["SE", "NO", "DK", "FI", "NL", "DE", "EE"],
    "mediterranean stucco housing": ["ES", "IT", "GR", "PT", "HR", "TR", "MT", "CY"],
    "soviet era concrete apartment blocks": ["RU", "UA", "BY", "KZ", "PL", "RO",
                                                "BG", "LT", "LV", "EE", "MD", "GE", "AM"],
    "east asian high rise housing": ["CN", "HK", "TW", "KR", "JP", "SG"],
    "south asian concrete housing": ["IN", "BD", "PK", "LK", "NP"],
    "latin american concrete housing": ["BR", "MX", "AR", "CO", "PE", "CL", "EC", "BO"],
    "sub saharan african housing": ["NG", "KE", "TZ", "GH", "ZA", "UG", "ET", "ZM"],
    "middle eastern flat roofed housing": ["EG", "SA", "AE", "JO", "IQ", "IR",
                                            "MA", "DZ", "TN", "LY", "IL", "PS"],
    "traditional japanese housing": ["JP"],
    "alpine chalet architecture": ["CH", "AT", "FR", "IT", "DE", "SI"],
}

# Hemisphere-sensitive seasonal cue: snow/bare trees imply high absolute
# latitude, and combined with a known month imply a hemisphere.
NORTHERN_WINTER_MONTHS = {11, 12, 1, 2, 3}
SOUTHERN_WINTER_MONTHS = {5, 6, 7, 8, 9}


# ---------------------------------------------------------------------------
# UIC rolling-stock numbers. Every vehicle on the European network carries a
# 12-digit number in which digits 3-4 are the keeper country -- so a legible
# train flank is a hard country identifier, not an inference. Standard rail
# OSINT tradecraft, and unambiguous where a shop name or a road sign is not.
#
# Example: "93 87 0029..." -> 93 = electric traction unit, 87 = France (SNCF).
# ---------------------------------------------------------------------------
UIC_COUNTRY: dict[str, str] = {
    "10": "FI", "20": "RU", "21": "BY", "22": "UA", "23": "MD", "24": "LT",
    "25": "LV", "26": "EE", "27": "KZ", "28": "GE", "29": "UZ", "30": "KP",
    "31": "MN", "32": "VN", "33": "CN", "40": "CU", "41": "AL", "42": "JP",
    "44": "BA", "49": "BA", "50": "BA", "51": "PL", "52": "BG", "53": "RO",
    "54": "CZ", "55": "HU", "56": "SK", "57": "AZ", "58": "AM", "59": "KG",
    "60": "IE", "61": "KR", "62": "ME", "63": "MK", "65": "MK", "66": "TJ",
    "67": "TM", "68": "AF", "70": "GB", "71": "ES", "72": "RS", "73": "GR",
    "74": "SE", "75": "TR", "76": "NO", "78": "HR", "79": "SI", "80": "DE",
    "81": "AT", "82": "LU", "83": "IT", "84": "NL", "85": "CH", "86": "DK",
    "87": "FR", "88": "BE", "90": "EG", "91": "TN", "92": "DZ", "93": "MA",
    "94": "PT", "95": "IL", "96": "IR", "97": "SY", "98": "LB", "99": "IQ",
}

# Leading two digits of a UIC number that indicate a self-propelled traction
# unit or coach rather than a freight wagon. Present only to sanity-check
# that a run of digits really is a vehicle number.
UIC_TYPE_PREFIXES: frozenset[str] = frozenset(
    # Traction units (locomotives, multiple units) are 90-99; hauled coaches
    # occupy 50-79. Freight wagons use 00-49 and 80-89, which are excluded:
    # they overlap too readily with prices and phone numbers for a partial,
    # unchecked number to be trusted.
    [str(n) for n in range(90, 100)] + [str(n) for n in range(50, 80)]
)

# Aircraft registration prefixes visible on tails and fuselages.
AIRCRAFT_PREFIX_COUNTRY: dict[str, str] = {
    "G-": "GB", "F-": "FR", "D-": "DE", "I-": "IT", "EC-": "ES", "PH-": "NL",
    "OO-": "BE", "LX-": "LU", "OE-": "AT", "HB-": "CH", "SE-": "SE",
    "LN-": "NO", "OY-": "DK", "OH-": "FI", "TF-": "IS", "EI-": "IE",
    "SP-": "PL", "OK-": "CZ", "OM-": "SK", "HA-": "HU", "YR-": "RO",
    "LZ-": "BG", "SX-": "GR", "TC-": "TR", "9A-": "HR", "S5-": "SI",
    "YU-": "RS", "Z3-": "MK", "CS-": "PT", "N": "US", "C-": "CA",
    "VH-": "AU", "ZK-": "NZ", "JA": "JP", "B-": "CN", "HL": "KR",
    "VT-": "IN", "PP-": "BR", "PR-": "BR", "LV-": "AR", "CC-": "CL",
    "ZS-": "ZA", "A6-": "AE", "A7-": "QA", "9V-": "SG", "9M-": "MY",
    "HS-": "TH", "PK-": "ID", "RP-": "PH", "4X-": "IL", "SU-": "EG",
}
