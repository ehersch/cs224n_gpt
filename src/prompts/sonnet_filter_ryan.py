import json
import re

import nltk
import pronouncing


def parse_sonnets(path):
    with open(path, "r") as f:
        text = f.read()

    lst = re.split("###.*###", text)
    return lst[1:]


def validate_sonnet(text):
    # 1. Clean and split into lines
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    results = {
        "line_count": len(lines) == 14,
        "meter_score": 0,
        "rhyme_valid": False,
        "errors": []
    }

    if not results["line_count"]:
        results["errors"].append(f"Expected 14 lines, found {len(lines)}")

    # 2. Check Iambic Pentameter (Simplified)
    # Ideally 10 syllables per line with an unstressed/stressed pattern
    meter_passed = 0
    for line in lines:
        words = nltk.tokenize.word_tokenize(line)
        # Get syllable count from pronouncing dictionary
        syllables = 0
        for word in words:
            phones = pronouncing.phones_for_word(word)
            if phones:
                syllables += pronouncing.syllable_count(phones[0])
        
        if 9 <= syllables <= 11: # Allow slight variation
            meter_passed += 1
    
    results["meter_score"] = meter_passed / 14.0

    # 3. Check Rhyme Scheme (ABAB CDCD EFEF GG)
    # We check if the last word of line i rhymes with line j
    def get_rhyme_part(word):
        phones = pronouncing.phones_for_word(word)
        return pronouncing.rhyming_part(phones[0]) if phones else None

    def rhymes(w1, w2):
        p1 = get_rhyme_part(w1.strip(".,!?;"))
        p2 = get_rhyme_part(w2.strip(".,!?;"))
        return p1 == p2 and p1 is not None

    # Shakespearean pairs: (0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)
    rhyme_pairs = [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]
    valid_rhymes = 0
    if len(lines) == 14:
        for p1, p2 in rhyme_pairs:
            w1 = lines[p1].split()[-1]
            w2 = lines[p2].split()[-1]
            if rhymes(w1, w2):
                valid_rhymes += 1
    
    results["rhyme_valid"] = valid_rhymes == len(rhyme_pairs)
    results["rhyme_score"] = valid_rhymes / len(rhyme_pairs)

    return results


def filter_sonnets(sonnet_lst):
    data = []
    for sonnet in sonnet_lst:
        results = validate_sonnet(sonnet)
        if results["line_count"] and results["rhyme_valid"]:
            data.append(sonnet)
    print(len(data))
    with open("synthetic_data/filtered_sonnets_2.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    return data


if __name__ == "__main__":
    lst = parse_sonnets("synthetic_data/synthetic_sonnets_2.txt")
    filter_sonnets(lst)