CATEGORIES = {
    "IT - System Access issue": {
        "strong": ["password reset", "id locked", "access granted"]
    },
    "User - Multiple versions issue in excel": {
        "strong": ["wrong excel version", "multiple excel files"]
    }
}

def rule_override(text):
    text = text.lower()
    for category, keywords in CATEGORIES.items():
        for word in keywords["strong"]:
            if word in text:
                return category
    return None
