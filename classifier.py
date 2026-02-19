# classifier.py

CATEGORIES = {

"IT - System Access issue": {
    "strong": ["access granted", "id unlocked", "role assigned"],
    "primary": ["login", "password", "unlock", "access denied", "role issue"],
    "context": ["authorization", "user id", "credentials"]
},

"IT - System linkage issue": {
    "strong": ["integration fixed", "interface corrected"],
    "primary": ["integration", "not syncing", "interface error"],
    "context": ["data not reflecting", "sync issue", "linkage"]
},

"IT – System Version issue": {
    "strong": ["version upgraded", "patch installed"],
    "primary": ["version mismatch", "old version"],
    "context": ["upgrade", "patch issue"]
},

"IT – Data entry handholding": {
    "strong": ["assisted user", "guided user"],
    "primary": ["data entry help", "support for entry"],
    "context": ["handholding", "helped enter data"]
},

"IT – Master Data/ mapping issue": {
    "strong": ["mapping corrected", "master updated"],
    "primary": ["mapping issue", "master data error"],
    "context": ["not mapped", "incorrect mapping"]
},

"User - Mapping missing": {
    "strong": ["user missed mapping"],
    "primary": ["mapping missing"],
    "context": ["not provided mapping"]
},

"User – Master data delayed input": {
    "strong": ["data provided late"],
    "primary": ["delayed input", "late submission"],
    "context": ["not uploaded on time"]
},

"User - Logic changes during ABP": {
    "strong": ["logic changed by user"],
    "primary": ["logic change", "abp change"],
    "context": ["requirement changed"]
},

"User – Master data incorporation in system": {
    "strong": ["data incorporated"],
    "primary": ["incorporation issue"],
    "context": ["not updated in system"]
},

"User – System Knowledge Gap": {
    "strong": ["user trained", "explained to user"],
    "primary": ["how to", "guidance needed"],
    "context": ["not aware", "clarification required"]
},

"User - Logic mistakes in excel vs system": {
    "strong": ["excel logic incorrect"],
    "primary": ["logic mismatch", "excel vs system"],
    "context": ["calculation difference"]
},

"User - Multiple versions issue in excel": {
    "strong": ["used wrong excel version"],
    "primary": ["multiple excel", "version mismatch"],
    "context": ["old file", "different sheet"]
}

}


def auto_label(text):
    text = text.lower()
    scores = {}

    for category, keywords in CATEGORIES.items():
        score = 0

        for word in keywords["strong"]:
            if word in text:
                score += 3

        for word in keywords["primary"]:
            if word in text:
                score += 2

        for word in keywords["context"]:
            if word in text:
                score += 1

        scores[category] = score

    return max(scores, key=scores.get)
