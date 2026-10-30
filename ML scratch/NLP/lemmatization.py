def lemmatize(word, pos="noun"):
    word = word.lower()

    # 1. Irregular forms: dictionary lookup
    irregular = {
        ("went", "verb"): "go",
        ("gone", "verb"): "go",
        ("was", "verb"): "be",
        ("were", "verb"): "be",
        ("is", "verb"): "be",
        ("am", "verb"): "be",
        ("are", "verb"): "be",
        ("has", "verb"): "have",
        ("had", "verb"): "have",
        ("did", "verb"): "do",
        ("done", "verb"): "do",
        ("better", "adjective"): "good",
        ("best", "adjective"): "good",
        ("mice", "noun"): "mouse",
        ("men", "noun"): "man",
        ("women", "noun"): "woman",
        ("children", "noun"): "child",
        ("feet", "noun"): "foot",
        ("teeth", "noun"): "tooth",
        ("saw", "verb"): "see",
    }

    if (word, pos) in irregular:
        return irregular[(word, pos)]

    # 2. Noun morphology
    if pos == "noun":
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("ves") and len(word) > 4:
            # A simplified rule; not valid for every word.
            return word[:-3] + "f"

        if word.endswith("s") and not word.endswith("ss"):
            if word.endswith(("shes", "ches", "xes", "zes", "sses")):
                return word[:-2]

            if len(word) > 3:
                return word[:-1]

        return word

    # 3. Verb morphology
    if pos == "verb":
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("ing") and len(word) > 5:
            base = word[:-3]

            # running -> run, stopping -> stop
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]

            # making -> make, writing -> write
            if base.endswith(("mak", "tak", "writ", "giv", "us")):
                base += "e"

            return base

        if word.endswith("ied") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("ed") and len(word) > 4:
            base = word[:-2]

            # stopped -> stop
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]

            # loved -> love, danced -> dance
            if base.endswith(("lov", "mov", "danc", "us", "writ")):
                base += "e"

            return base

        if word.endswith("es") and len(word) > 4:
            if word.endswith(("ches", "shes", "xes", "zes", "sses")):
                return word[:-2]

        if word.endswith("s") and len(word) > 3:
            return word[:-1]

        return word

    # 4. Adjective morphology
    if pos == "adjective":
        if word.endswith("iest") and len(word) > 5:
            return word[:-4] + "y"

        if word.endswith("est") and len(word) > 4:
            return word[:-3]

        if word.endswith("er") and len(word) > 4:
            return word[:-2]

        return word

    # Unknown POS: don't guess
    return word


examples = [
    ("studies", "noun"),
    ("mice", "noun"),
    ("playing", "verb"),
    ("running", "verb"),
    ("stopped", "verb"),
    ("went", "verb"),
    ("better", "adjective"),
    ("saw", "verb"),
    ("saw", "noun"),
]

for word, pos in examples:
    print(f"{word:12} {pos:12} -> {lemmatize(word, pos)}")