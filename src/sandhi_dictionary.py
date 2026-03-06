"""
Expanded Sandhi Dictionary for Malayalam

This file contains comprehensive sandhi rules and compound word splits
for Malayalam, compiled from Wikipedia, linguistic research, and the SMC corpus.

Categories:
1. Place Names (സ്ഥലനാമങ്ങൾ)
2. Compound Nouns (സമാസപദങ്ങൾ)
3. Sandhi Types and Rules
4. Common Transformations
"""

# ============================================================================
# PLACE NAMES (സ്ഥലനാമങ്ങൾ)
# Most Kerala place names are compound words formed by sandhi
# ============================================================================

PLACE_NAMES = {
    # District headquarters
    "തിരുവനന്തപുരം": {
        "split": ["തിരു", "അനന്ത", "പുരം"],
        "meaning": "City of Lord Ananta (sacred + Ananta + city)",
        "sandhi_type": "സ്വരസന്ധി"
    },
    "കൊല്ലം": {
        "split": ["കൊല്ലം"],
        "meaning": "Quilon (single word)",
        "sandhi_type": None
    },
    "പത്തനംതിട്ട": {
        "split": ["പത്തനം", "തിട്ട"],
        "meaning": "Array of houses (houses + array)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ആലപ്പുഴ": {
        "split": ["ആല", "പ്പുഴ"],
        "meaning": "Between rivers/banyan + river",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കോട്ടയം": {
        "split": ["കോട്ട", "യം"],
        "meaning": "Inside fort (fort + inside)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ഇടുക്കി": {
        "split": ["ഇടു", "ക്കി"],
        "meaning": "Gorge/ravine",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "പാലക്കാട്": {
        "split": ["പാല", "കാട്"],
        "meaning": "Forest of milk (milk + forest)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "തൃശ്ശൂർ": {
        "split": ["തൃശ്ശൂർ"],
        "meaning": "Thrissur (shortened form)",
        "sandhi_type": None
    },
    "എറണാകുളം": {
        "split": ["എറണ", "കുളം"],
        "meaning": "Pond of Erayana (name + pond)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കണ്ണൂർ": {
        "split": ["കണ്ണൻ", "ഊര്"],
        "meaning": "City of Kannan (Krishna's city)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "വയനാട്": {
        "split": ["വയ", "നാട്"],
        "meaning": "Land of paddy fields",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "മലപ്പുറം": {
        "split": ["മല", "പ്പുറം"],
        "meaning": "Hilly region (hill + other side)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കോഴിക്കോട്": {
        "split": ["കോഴി", "ക്കോട്"],
        "meaning": "Chicken fort/coast",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കാസർഗോഡ്": {
        "split": ["കാസർ", "ഗോഡ്"],
        "meaning": "Fort of Kaisaru",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    
    # Other important places
    "കൊച്ചി": {
        "split": ["കൊച്ചി"],
        "meaning": "Cochin",
        "sandhi_type": None
    },
    "മാവേലിക്കര": {
        "split": ["മാവേലി", "ക്കര"],
        "meaning": "Land of Mahabali",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കോട്ടയം": {
        "split": ["കോട്ട", "യം"],
        "meaning": "Inside the fort",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ചേർത്തല": {
        "split": ["ചേർ", "തല"],
        "meaning": "Joined head/place",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ആറന്മുള": {
        "split": ["ആറൻ", "മുള"],
        "meaning": "Six bamboos",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ശബരിമല": {
        "split": ["ശബരി", "മല"],
        "meaning": "Mountain of Sabari",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ഗുരുവായൂർ": {
        "split": ["ഗുരു", "വായൂർ"],
        "meaning": "Place of Guru (teacher/wind)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "വൈക്കം": {
        "split": ["വൈ", "ക്കം"],
        "meaning": "Place name",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കുമരകം": {
        "split": ["കുമാര", "കം"],
        "meaning": "Place of Kumar/Kartikeya",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "തേക്കടി": {
        "split": ["തേക്ക്", "അടി"],
        "meaning": "Low land in south",
        "sandhi_type": "സ്വരസന്ധി"
    },
    "സൈലന്റ് വാലി": {
        "split": ["സൈലന്റ്", "വാലി"],
        "meaning": "Silent Valley (English loan)",
        "sandhi_type": None
    },
}

# ============================================================================
# COMPOUND NOUNS (സമാസപദങ്ങൾ)
# These are common compound words formed by various types of sandhi
# ============================================================================

COMPOUND_NOUNS = {
    # Educational institutions
    "വിദ്യാലയം": {
        "split": ["വിദ്യ", "ആലയം"],
        "meaning": "School (knowledge + abode)",
        "sandhi_type": "സ്വരസന്ധി"
    },
    "പാഠശാല": {
        "split": ["പാഠ", "ശാല"],
        "meaning": "School (lesson + hall)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ഗ്രന്ഥശാല": {
        "split": ["ഗ്രന്ഥ", "ശാല"],
        "meaning": "Library (book + hall)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "പത്രാധിപർ": {
        "split": ["പത്ര", "അധിപർ"],
        "meaning": "Editor (newspaper + lord)",
        "sandhi_type": "സ്വരസന്ധി"
    },
    
    # Government/Administrative
    "പ്രധാനമന്ത്രി": {
        "split": ["പ്രധാന", "മന്ത്രി"],
        "meaning": "Prime Minister (chief + minister)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "രാഷ്ട്രപതി": {
        "split": ["രാഷ്ട്ര", "പതി"],
        "meaning": "President (nation + lord)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "മുഖ്യമന്ത്രി": {
        "split": ["മുഖ്യ", "മന്ത്രി"],
        "meaning": "Chief Minister",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "പൊതുജനം": {
        "split": ["പൊതു", "ജനം"],
        "meaning": "Public (common + people)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "സർക്കാർ": {
        "split": ["സർക്കാർ"],
        "meaning": "Government (loan word)",
        "sandhi_type": None
    },
    
    # Medical/Health
    "ആയുർവേദം": {
        "split": ["ആയുർ", "വേദം"],
        "meaning": "Ayurveda (life + knowledge)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "രക്തസമ്മർദ്ദം": {
        "split": ["രക്ത", "സമ്മർദ്ദം"],
        "meaning": "Blood pressure",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "ഹൃദ്രോഗം": {
        "split": ["ഹൃദ്", "രോഗം"],
        "meaning": "Heart disease",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "അസുഖം": {
        "split": ["അ", "സുഖം"],
        "meaning": "Disease (not + health)",
        "sandhi_type": "സ്വരസന്ധി"
    },
    
    # Cultural/Religious
    "ക്ഷേത്രം": {
        "split": ["ക്ഷേത്രം"],
        "meaning": "Temple/field",
        "sandhi_type": None
    },
    "ദേവാലയം": {
        "split": ["ദേവ", "ആലയം"],
        "meaning": "Temple (god + abode)",
        "sandhi_type": "സ്വരസന്ധി"
    },
    "പള്ളി": {
        "split": ["പള്ളി"],
        "meaning": "Mosque/church",
        "sandhi_type": None
    },
    "മഠം": {
        "split": ["മഠം"],
        "meaning": "Monastery",
        "sandhi_type": None
    },
    
    # Body parts compounds
    "തലമുടി": {
        "split": ["തല", "മുടി"],
        "meaning": "Hair (head + hair)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കണ്ണീർ": {
        "split": ["കണ്ണ്", "നീർ"],
        "meaning": "Tears (eye + water)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കൈവിരൽ": {
        "split": ["കൈ", "വിരൽ"],
        "meaning": "Finger (hand + finger)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    "കാല്മുട്ട്": {
        "split": ["കാൽ", "മുട്ട്"],
        "meaning": "Knee (leg + knee)",
        "sandhi_type": "വ്യഞ്ജനസന്ധി"
    },
    
    # Common objects/concepts
    "ഭക്ഷണം": {
        "split": ["ഭക്ഷ", "അണം"],
        "meaning": "Food",
        "sandhi_type": "സ്വരസന്ധി"
    },
    "വാഹനം": {
        "split": ["വാഹനം"],
        "meaning": "Vehicle",
        "sandhi_type": None
    },
    "വീട്": {
        "split": ["വീട്"],
        "meaning": "House",
        "sandhi_type": None
    },
    "വഴി": {
        "split": ["വഴി"],
        "meaning": "Way/road",
        "sandhi_type": None
    },
}

# ============================================================================
# SANDHI TYPES AND RULES
# സന്ധി വിവരണം
# ============================================================================

SANDHI_RULES = {
    "സ്വരസന്ധി": {
        "description": "Vowel Sandhi - when two vowels meet",
        "rules": [
            {
                "name": "അ-അ സന്ധി",
                "pattern": "അ + അ → ആ",
                "examples": ["വിദ്യ + ആലയം = വിദ്യാലയം"]
            },
            {
                "name": "അ-ആ സന്ധി",
                "pattern": "അ + ആ → ആ",
                "examples": ["രാമ + ആയനം = രാമായനം"]
            },
            {
                "name": "അ-ഇ സന്ധി",
                "pattern": "അ + ഇ → എ (lopa)",
                "examples": ["പഠിക്ക + ഇൻ = പഠിക്കെൻ"]
            },
            {
                "name": "ഇ-അ സന്ധി",
                "pattern": "ഇ + അ → എ (vriddhi)",
                "examples": []
            },
            {
                "name": "ഉ-അ സന്ധി",
                "pattern": "ഉ + അ → ഒ",
                "examples": []
            },
        ]
    },
    
    "വ്യഞ്ജനസന്ധി": {
        "description": "Consonant Sandhi - when consonants meet",
        "rules": [
            {
                "name": "ലോപം (Elision)",
                "pattern": "Final vowel is dropped",
                "examples": ["പഠിക്ക + ുന്നു = പഠിക്കുന്നു"]
            },
            {
                "name": "ദ്വിത്വം (Doubling)",
                "pattern": "Consonant is doubled",
                "examples": ["പഠി + ക + ുന്നു = പഠിക്കുന്നു"]
            },
            {
                "name": "ആഗമം (Insertion)",
                "pattern": "Sound is inserted",
                "examples": ["പാല + കാട് = പാലക്കാട് (y insertion)"]
            },
        ]
    },
    
    "വിസർഗ്ഗസന്ധി": {
        "description": "Visarga Sandhi - involving ഃ",
        "rules": [
            {
                "name": "വിസർഗ്ഗലോപം",
                "pattern": "Visarga is dropped",
                "examples": []
            }
        ]
    },
    
    "ലോപസന്ധി": {
        "description": "Elision - sounds are dropped",
        "rules": [
            {
                "name": "അ-ലോപം",
                "pattern": "Final 'a' is dropped before vowel",
                "examples": ["പഠിക്ക → പഠിക്ക് before ുന്നു"]
            },
        ]
    }
}

# ============================================================================
# TRANSFORMATION PATTERNS
# Common patterns for sandhi transformation
# ============================================================================

TRANSFORMATION_PATTERNS = {
    # Past tense transformations (verb class changes)
    "past_tense": {
        "ക്ക് → ച്ച്": {
            "examples": ["പഠിക്കുക → പഠിച്ചു", "നടക്കുക → നടന്നു"],
            "description": "Transitivizer ക്ക changes to ച്ച in past"
        },
        "ക്ക് → ത്ത്": {
            "examples": ["എടുക്കുക → എടുത്തു", "കൊടുക്കുക → കൊടുത്തു"],
            "description": "Some ക്ക verbs change to ത്ത in past"
        },
        "് → ി": {
            "examples": ["വരുക → വന്നു", "പോകുക → പോയി"],
            "description": "Irregular past forms"
        }
    },
    
    # Noun case transformations
    "case_markers": {
        "ം → ത്ത്": {
            "before_vowel": True,
            "examples": ["വിദ്യാലയം + ിൽ → വിദ്യാലയത്തിൽ"],
            "description": "Nouns ending in ം change to ത്ത് before vowel-initial case markers"
        },
        "ം → ട്ട്": {
            "before_vowel": True,
            "examples": [],
            "description": "Some nouns change ം to ട്ട്"
        }
    },
    
    # Vowel insertions
    "vowel_insertion": {
        "അ insertion": {
            "trigger": "Consonant-initial suffix after stem",
            "examples": ["പഠിക്ക് + ണം → പഠിക്കണം"],
            "description": "Insert അ between stem ending in ് and consonant-initial suffix"
        }
    }
}

# ============================================================================
# HIGH FREQUENCY COMPOUNDS FROM SMC CORPUS
# Words that frequently appear in the corpus
# ============================================================================

HIGH_FREQUENCY_COMPOUNDS = {
    "സ്വാതന്ത്ര്യം": ["സ്വ", "തന്ത്ര്യം"],
    "പ്രധാനമന്ത്രി": ["പ്രധാന", "മന്ത്രി"],
    "വിദ്യാഭ്യാസം": ["വിദ്യ", "അഭ്യാസം"],
    "സർക്കാർ": ["സർക്കാർ"],
    "വികസനം": ["വികസനം"],
    "പ്രവർത്തനം": ["പ്രവർത്ത", "നം"],
    "പരിപാടി": ["പരി", "പാടി"],
    "സംഘടന": ["സംഘ", "ടന"],
    "പൊതുജനം": ["പൊതു", "ജനം"],
    "കാര്യദർശി": ["കാര്യ", "ദർശി"],
    "പഞ്ചായത്ത്": ["പഞ്ച", "ആയത്ത്"],
    "നിയമസഭ": ["നിയമ", "സഭ"],
    "പാർലമെന്റ്": ["പാർലമെന്റ്"],
    "സാംസ്കാരികം": ["സാംസ്കാരികം"],
    "വിപണനം": ["വിപണ", "നം"],
}


def get_all_compounds():
    """Return all compound words from all categories."""
    all_compounds = {}
    all_compounds.update(PLACE_NAMES)
    all_compounds.update(COMPOUND_NOUNS)
    all_compounds.update(HIGH_FREQUENCY_COMPOUNDS)
    return all_compounds


def lookup_compound(word: str):
    """Look up a word in the sandhi dictionary."""
    all_compounds = get_all_compounds()
    return all_compounds.get(word)


def get_split(word: str):
    """Get the split for a compound word."""
    entry = lookup_compound(word)
    if entry:
        # Handle both dict format and list format
        if isinstance(entry, dict):
            return entry.get("split", [word])
        elif isinstance(entry, list):
            return entry
    return [word]
