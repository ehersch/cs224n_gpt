from ast import parse
from pathlib import Path

import pronouncing
import re
import json

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_SYNTHETIC = REPO_ROOT / "data" / "synthetic_data"


def parse_sonnets(path):
    with open(path, "r") as f:
        text = f.read()

    lst = re.split("###.*###", text)
    return lst[1:]


def has_rhyme(sonnet):
    """
    scheme: ABAB CDCD EFEF GG
    """
    end_words = []
    for line in sonnet.splitlines():
        word = (line.split(" ")[-1]).strip()
        clean_word = re.sub(r"[,\.;'\"`!?]", "", word)
        if word != "":
            end_words += [clean_word]

    a = set(pronouncing.rhymes(end_words[0]))
    b = set(pronouncing.rhymes(end_words[1]))
    c = set(pronouncing.rhymes(end_words[4]))
    d = set(pronouncing.rhymes(end_words[5]))
    e = set(pronouncing.rhymes(end_words[8]))
    f = set(pronouncing.rhymes(end_words[9]))
    g = set(pronouncing.rhymes(end_words[12]))

    rhymes = (
        (end_words[2] in a)
        and (end_words[3] in b)
        and (end_words[6] in c)
        and (end_words[7] in d)
        and (end_words[10] in e)
        and (end_words[11] in f)
        and (end_words[13] in g)
    )
    return rhymes


def filter_sonnets(sonnet_lst):
    data = []
    for sonnet in sonnet_lst:
        if has_rhyme(sonnet):
            data += [sonnet]
    print(len(data))
    out_path = DATA_SYNTHETIC / "filtered_sonnets_2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return data


# with open("data/synthetic_sonnets.txt", 'r') as f:
#   for line in f

if __name__ == "__main__":
    lst = parse_sonnets(str(DATA_SYNTHETIC / "synthetic_sonnets_2.txt"))
    filter_sonnets(lst)
