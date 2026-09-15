"""
car_enrichment.py

Module for enriching used-car datasets with:
1. General listing flags (CarFax, Accident, One Owner, etc.) -> binary 0/1 columns
2. Engine/powertrain description extracted from the 'trim' column
3. Package/trim-level description extracted from the 'trim' column

All matching is case-insensitive and tolerant of spacing/dash/typo variations
(e.g. "350h" / "350 H" / "350-h" all match; "NOACCCIDENT" / "no-accident" /
"accident free" all flag as accident-free).

Usage in a notebook:
    import car_enrichment as ce
    df = ce.enrich_dataframe(df)
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. GENERAL FLAGS (make-independent)
#    Each flag maps to a compiled regex (not plain keywords) so spacing,
#    dashes, squished-together words, and common typos are all tolerated.
# ---------------------------------------------------------------------------
GENERAL_PATTERNS = {
    # tolerates: no accident / no-accident / noaccident / NOACCCIDENT /
    #            accident free / accidentfree
    "no_accidents": re.compile(r"\b(no[\s\-]*a+c+[iy]?dents?|a+c+[iy]?dents?[\s\-]*free)\b"),
    "has_carfax":   re.compile(r"\bcar[\s\-]*fax\b"),
    "one_owner":    re.compile(r"\b(one|1)[\s\-]*owner\b"),
    "low_mileage_mention": re.compile(r"\blow[\s\-]*(mileage|kms?|km)\b"),
    "service_records": re.compile(r"\bservice[\s\-]*(record|d)\b"),
    "certified":    re.compile(r"\bcertified\b"),
}


# ---------------------------------------------------------------------------
# 2. BRAND-SPECIFIC PACKAGE / ENGINE REFERENCE
# ---------------------------------------------------------------------------
BRAND_REFERENCE = {
    "BMW": {
        "packages": ["M Sport", "M Sport Pro", "Luxury Line", "Sport Line", "xLine",
                     "Premium Package", "Executive Package", "M Performance", "Individual"],
        "engine_names": ["xDrive", "sDrive", "M50", "M60", "40i", "30i", "20i", "M240i",
                          "M340i", "M440i", "M550i", "M760i", "Competition",
                          "M2", "M3", "M4", "M5", "M8"]
    },
    "Audi": {
        "packages": ["S line", "Black Edition", "Prestige", "Progressiv", "Komfort",
                     "Technik", "Titanium Black"],
        "engine_names": ["Quattro", "TFSI", "TDI", "40", "45", "50", "55", "60",
                          "S3", "S4", "S5", "S6", "S7", "S8",
                          "RS3", "RS5", "RS6", "RS7", "RSQ8"]
    },
    "Mercedes-Benz": {
        "packages": ["AMG Line", "Night Package", "Premium Package",
                     "Intelligent Drive Package", "Sport Package"],
        # Numbers matter a lot here (E450, C43, GLC63, etc.) so plain badge numbers
        # are included on their own in addition to the "NN AMG" phrasing.
        "engine_names": ["4MATIC", "AMG", "43", "45", "53", "63",
                          "43 AMG", "45 AMG", "53 AMG", "63 AMG",
                          "300", "350", "400", "450", "500", "580", "EQ Boost"]
    },
    "Volvo": {
        "packages": ["R-Design", "Inscription", "Momentum", "Polestar Engineered"],
        "engine_names": ["T4", "T5", "T6", "T8", "B5", "B6", "Recharge", "Twin Engine"]
    },
    "Land Rover": {
        "packages": ["HSE", "Autobiography", "First Edition", "SE", "Dynamic SE",
                     "Westminster Edition"],
        "engine_names": ["P250", "P300", "P360", "P400", "P400e", "P440e", "P460",
                          "P525", "P530", "D200", "D250", "D300", "SDV6"]
    },
    "Porsche": {
        "packages": ["Sport Chrono Package", "Premium Package", "Sport Design Package"],
        "engine_names": ["Carrera", "Carrera S", "Carrera 4S", "GTS", "Turbo", "Turbo S",
                          "GT3", "GT4", "4S", "Targa"]
    },
    "Lexus": {
        "packages": ["F Sport", "Luxury Package", "Executive Package", "Premium Package"],
        # Lexus badges: 350h/300h/450h/500h etc. - numbers + trailing hybrid letter matter
        "engine_names": ["200", "250", "300", "350", "430", "460", "500",
                          "200h", "250h", "300h", "350h", "450h", "500h", "F"]
    },
    "Infiniti": {
        "packages": ["Sport Package", "Premium Package", "ProACTIVE Package",
                     "Essential Package"],
        "engine_names": ["VR30", "3.0t", "AWD", "Pure", "Luxe", "Sensory", "Autograph"]
    },
    "Cadillac": {
        "packages": ["Luxury Package", "Premium Luxury", "Sport Package", "Platinum"],
        "engine_names": ["V-Series", "Blackwing", "V6", "V8", "Twin Turbo"]
    },
    "Genesis": {
        "packages": ["Advanced Package", "Prestige Package", "Ultimate Package"],
        "engine_names": ["3.5T", "2.5T", "V6", "V8"]
    },
    "Jaguar": {
        "packages": ["R-Dynamic", "Portfolio", "First Edition", "HSE"],
        "engine_names": ["P250", "P300", "P340", "P400", "P450", "SVR", "R-Dynamic S"]
    },
    "Maserati": {
        "packages": ["GranSport", "GranLusso", "Trofeo"],
        "engine_names": ["V6", "V8", "Modena", "Trofeo"]
    },
    "Bentley": {
        "packages": ["Mulliner", "First Edition"],
        "engine_names": ["V8", "W12", "Speed"]
    },
    "Rolls-Royce": {
        "packages": ["Black Badge"],
        "engine_names": ["V12"]
    },
    "Aston Martin": {
        "packages": ["Vantage", "AMR"],
        "engine_names": ["V8", "V12"]
    },
    "McLaren": {
        "packages": [],
        "engine_names": ["V8 Twin Turbo"]
    },
    "MINI": {
        "packages": ["Cooper S", "John Cooper Works", "JCW", "Chili Package"],
        "engine_names": ["Cooper", "Cooper S", "Cooper SE", "JCW"]
    },
    "Toyota": {
        "packages": ["TRD", "TRD Sport", "TRD Off-Road", "Limited", "Platinum", "XLE",
                     "XSE", "SE", "LE", "Nightshade Edition"],
        "engine_names": ["Hybrid", "Prime", "AWD-e", "i-FORCE MAX"]
    },
    "Honda": {
        "packages": ["Touring", "EX-L", "EX", "LX", "Sport", "Black Edition", "TrailSport"],
        "engine_names": ["Hybrid", "e:HEV", "VTEC Turbo"]
    },
    "Ford": {
        "packages": ["Platinum", "King Ranch", "Lariat", "XLT", "XL", "ST-Line",
                     "Limited", "Tremor", "Raptor"],
        "engine_names": ["EcoBoost", "PowerBoost Hybrid", "Hybrid"]
    },
    "Chevrolet": {
        "packages": ["RST", "LT", "LTZ", "High Country", "Premier", "Z71", "SS"],
        "engine_names": ["Turbo", "V6", "V8", "eAssist"]
    },
    "GMC": {
        "packages": ["Denali", "AT4", "SLE", "SLT", "Elevation"],
        "engine_names": ["Duramax", "Turbo"]
    },
    "RAM": {
        "packages": ["Laramie", "Rebel", "Big Horn", "Limited", "TRX"],
        "engine_names": ["HEMI", "EcoDiesel", "Cummins"]
    },
    "Dodge": {
        "packages": ["R/T", "SRT", "GT", "SXT"],
        "engine_names": ["HEMI", "SRT Hellcat", "V6", "V8"]
    },
    "Jeep": {
        "packages": ["Trailhawk", "Rubicon", "Sahara", "Limited", "Overland", "Summit",
                     "Willys", "High Altitude"],
        "engine_names": ["4xe", "Hurricane", "eTorque"]
    },
    "Subaru": {
        "packages": ["Limited", "Touring", "Premium", "Sport", "Wilderness", "STI"],
        "engine_names": ["Boxer", "Turbo", "H4", "H6"]
    },
    "Mazda": {
        "packages": ["GT", "Signature", "Premium", "Preferred", "Turbo"],
        "engine_names": ["Skyactiv", "Skyactiv-X", "Turbo"]
    },
    "Hyundai": {
        "packages": ["Ultimate", "Luxury", "Preferred", "Essential", "N Line", "Calligraphy"],
        "engine_names": ["Turbo", "Hybrid", "N Line", "N"]
    },
    "Kia": {
        "packages": ["SX", "SX Prestige", "EX", "GT-Line", "EX Premium"],
        "engine_names": ["Turbo", "GT", "Hybrid"]
    },
    "Nissan": {
        "packages": ["Platinum", "SL", "SV", "SR", "Midnight Edition"],
        "engine_names": ["VC-Turbo", "Nismo"]
    },
    "Volkswagen": {
        "packages": ["Highline", "Comfortline", "Trendline", "Execline", "R-Line"],
        "engine_names": ["TSI", "TDI", "4MOTION", "GTI", "R"]
    },
    "Acura": {
        "packages": ["A-Spec", "Advance", "Elite", "Type S"],
        "engine_names": ["SH-AWD", "Turbo", "Type S"]
    },
    "Buick": {
        "packages": ["Avenir", "Essence", "Premium"],
        "engine_names": ["Turbo"]
    },
    "Chrysler": {
        "packages": ["Limited", "Touring", "Pinnacle"],
        "engine_names": ["V6"]
    },
    "Tesla": {
        "packages": ["Long Range", "Performance", "Plaid", "Standard Range"],
        "engine_names": ["Dual Motor", "Single Motor", "Tri Motor"]
    },
    "Mitsubishi": {
        "packages": ["GT", "SE", "ES", "Limited"],
        "engine_names": ["Turbo"]
    },
    "smart": {
        "packages": [],
        "engine_names": []
    },
}


# ---------------------------------------------------------------------------
# 3. COUNTRY OF ORIGIN / BRAND SEGMENT LOOKUP
#    Country of origin is a stable fact; segment (Economy/Mainstream/Luxury/
#    Ultra-luxury/Exotic) is a reasonable industry-standard classification,
#    not an objective fact - adjust as needed for your use case.
# ---------------------------------------------------------------------------
COUNTRY_SEGMENT_LOOKUP = {
    "Lincoln": {"country": "USA", "segment": "Luxury"},
    "Jeep": {"country": "USA", "segment": "Mainstream"},
    "Hyundai": {"country": "South Korea", "segment": "Economy"},
    "Mazda": {"country": "Japan", "segment": "Mainstream"},
    "Ford": {"country": "USA", "segment": "Mainstream"},
    "Land Rover": {"country": "UK", "segment": "Luxury"},
    "Honda": {"country": "Japan", "segment": "Mainstream"},
    "Kia": {"country": "South Korea", "segment": "Economy"},
    "Genesis": {"country": "South Korea", "segment": "Luxury"},
    "Audi": {"country": "Germany", "segment": "Luxury"},
    "Subaru": {"country": "Japan", "segment": "Mainstream"},
    "Toyota": {"country": "Japan", "segment": "Mainstream"},
    "BMW": {"country": "Germany", "segment": "Luxury"},
    "Infiniti": {"country": "Japan", "segment": "Luxury"},
    "Nissan": {"country": "Japan", "segment": "Mainstream"},
    "Mercedes-Benz": {"country": "Germany", "segment": "Luxury"},
    "Volkswagen": {"country": "Germany", "segment": "Mainstream"},
    "Porsche": {"country": "Germany", "segment": "Luxury"},
    "Buick": {"country": "USA", "segment": "Near-luxury"},
    "Acura": {"country": "Japan", "segment": "Luxury"},
    "MINI": {"country": "UK", "segment": "Premium"},
    "Tesla": {"country": "USA", "segment": "Luxury"},
    "Lexus": {"country": "Japan", "segment": "Luxury"},
    "RAM": {"country": "USA", "segment": "Mainstream"},
    "Chevrolet": {"country": "USA", "segment": "Economy"},
    "GMC": {"country": "USA", "segment": "Near-luxury"},
    "Volvo": {"country": "Sweden", "segment": "Luxury"},
    "Mitsubishi": {"country": "Japan", "segment": "Economy"},
    "Dodge": {"country": "USA", "segment": "Mainstream"},
    "Chrysler": {"country": "USA", "segment": "Mainstream"},
    "Cadillac": {"country": "USA", "segment": "Luxury"},
    "Scion": {"country": "Japan", "segment": "Economy"},
    "Rivian": {"country": "USA", "segment": "Luxury"},
    "Alfa": {"country": "Italy", "segment": "Luxury"},
    "Jaguar": {"country": "UK", "segment": "Luxury"},
    "Fiat": {"country": "Italy", "segment": "Economy"},
    "Saturn": {"country": "USA", "segment": "Economy"},
    "Renault": {"country": "France", "segment": "Economy"},
    "Pontiac": {"country": "USA", "segment": "Mainstream"},
    "Maserati": {"country": "Italy", "segment": "Ultra-luxury"},
    "Fisker": {"country": "USA", "segment": "Luxury"},
    "Ferrari": {"country": "Italy", "segment": "Exotic"},
    "Aston": {"country": "UK", "segment": "Exotic"},
    "smart": {"country": "Germany", "segment": "Economy"},
    "Rolls-Royce": {"country": "UK", "segment": "Ultra-luxury"},
    "Lotus": {"country": "UK", "segment": "Exotic"},
    "Polestar": {"country": "Sweden", "segment": "Luxury"},
    "Bentley": {"country": "UK", "segment": "Ultra-luxury"},
    "Saab": {"country": "Sweden", "segment": "Near-luxury"},
    "McLaren": {"country": "UK", "segment": "Exotic"},
    "Suzuki": {"country": "Japan", "segment": "Economy"},
    "HUMMER": {"country": "USA", "segment": "Luxury"},
    "Daihatsu": {"country": "Japan", "segment": "Economy"},
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _flexible_term_pattern(term):
    """Build a regex for `term` that tolerates:
       - space / dash / no-separator between its words
         ('m sport' matches 'M-Sport' / 'MSport' / 'M Sport')
       - space / dash / no-separator at digit<->letter boundaries
         ('350h' matches '350 H' / '350-h' / '350H')
       - a letter directly touching the term from outside
         ('450' matches inside 'E450', '43' matches inside 'GLC43'),
         while still NOT matching '450' inside '34500' or '4501'.
    """
    escaped = re.escape(term.lower())
    escaped = re.sub(r"\\ |\\-", lambda m: r"[\s\-]*", escaped)
    escaped = re.sub(r"(?<=\d)(?=[a-z])", lambda m: r"[\s\-]?", escaped)
    escaped = re.sub(r"(?<=[a-z])(?=\d)", lambda m: r"[\s\-]?", escaped)

    left = r"(?<!\d)" if term[0].isdigit() else r"\b"
    right = r"(?!\d)" if term[-1].isdigit() else r"\b"
    return left + escaped + right


def _find_brand_matches(make, trim_text, category):
    """Return list of matching packages/engine names for a given make."""
    if pd.isna(trim_text) or make not in BRAND_REFERENCE:
        return []
    trim_lower = str(trim_text).lower()
    candidates = BRAND_REFERENCE[make].get(category, [])
    matches = []
    for c in candidates:
        pattern = _flexible_term_pattern(c)
        if re.search(pattern, trim_lower):
            matches.append(c)
    return matches


def _general_flags(trim_text):
    """Return dict of {flag_name: 0/1} for a single trim string."""
    if pd.isna(trim_text):
        return {name: 0 for name in GENERAL_PATTERNS}
    text_lower = str(trim_text).lower()
    return {
        name: int(bool(pattern.search(text_lower)))
        for name, pattern in GENERAL_PATTERNS.items()
    }


def add_country_and_segment(df, make_col="Make"):
    """
    Adds 'Country_of_Origin' and 'Brand_Segment' columns to df, based on
    COUNTRY_SEGMENT_LOOKUP. Rows whose make isn't in the lookup get None
    for both columns.
    """
    df["Country_of_Origin"] = df[make_col].map(
        lambda x: COUNTRY_SEGMENT_LOOKUP.get(x, {}).get("country")
    )
    df["Brand_Segment"] = df[make_col].map(
        lambda x: COUNTRY_SEGMENT_LOOKUP.get(x, {}).get("segment")
    )
    return df


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def enrich_dataframe(df, make_col="Make", trim_col="trim"):
    """
    Adds the following columns to df:
        - one 0/1 column per GENERAL_PATTERNS entry
          (has_carfax, no_accidents, one_owner, low_mileage_mention,
           service_records, certified)
        - 'Engine_Description': matched engine/powertrain terms, joined as a string
        - 'Package_Description': matched package/trim terms, joined as a string
        - 'Country_of_Origin': the make's country of origin
        - 'Brand_Segment': the make's market segment (Economy/Mainstream/
          Luxury/Near-luxury/Ultra-luxury/Exotic/Premium)

    Parameters
    ----------
    df : pandas.DataFrame
    make_col : str, column name holding the car's make
    trim_col : str, column name holding the free-text trim/listing string

    Returns
    -------
    pandas.DataFrame (same object, modified in place and returned for chaining)
    """
    # --- General flags ---
    flags_df = df[trim_col].apply(_general_flags).apply(pd.Series)
    for col in flags_df.columns:
        df[col] = flags_df[col]

    # --- Engine description ---
    engine_matches = df.apply(
        lambda row: _find_brand_matches(row[make_col], row[trim_col], "engine_names"),
        axis=1
    )
    df["Engine_Description"] = engine_matches.apply(lambda lst: ", ".join(lst) if lst else None)

    # --- Package description ---
    package_matches = df.apply(
        lambda row: _find_brand_matches(row[make_col], row[trim_col], "packages"),
        axis=1
    )
    df["Package_Description"] = package_matches.apply(lambda lst: ", ".join(lst) if lst else None)

    # --- Country of origin / brand segment ---
    df = add_country_and_segment(df, make_col=make_col)

    return df