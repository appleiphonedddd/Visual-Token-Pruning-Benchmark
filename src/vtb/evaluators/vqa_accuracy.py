"""VQA accuracy (VQAv2 / TextVQA / VizWiz style).

Answer normalisation is the EvalAI processor used by the official VQA evaluation
and by LLaVA's ``m4c_evaluator.py``. Accuracy follows the official
leave-one-out rule: for each of the 10 human answers, score
min(1, #matches among the other 9 / 3), averaged over the 10.
"""

from __future__ import annotations

import re

from vtb.core.registry import EVALUATORS
from vtb.evaluators.base import BaseEvaluator

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've",
    "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
PUNCTUATIONS = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _process_punctuation(text: str) -> str:
    out = text
    for p in PUNCTUATIONS:
        if (p + " " in text or " " + p in text) or re.search(COMMA_STRIP, text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    # the reference implementation passes re.UNICODE (=32) as `count`; kept for identical results
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def _process_digit_article(text: str) -> str:
    words = []
    for w in text.lower().split():
        w = NUMBER_MAP.get(w, w)
        if w not in ARTICLES:
            words.append(w)
    return " ".join(CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(ans: str) -> str:
    ans = ans.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    return _process_digit_article(_process_punctuation(ans))


def vqa_score(pred: str, gt_answers: list[str]) -> float:
    gts = [normalize_answer(a) for a in gt_answers]
    p = normalize_answer(pred)
    if len(gts) == 1:  # single-reference datasets: plain exact match after normalisation
        return float(p == gts[0])
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3))
    return sum(accs) / len(accs)


@EVALUATORS.register("vqa_accuracy")
class VQAAccuracy(BaseEvaluator):
    primary_metric = "accuracy"

    def score(self, records: list[dict]) -> dict[str, float]:
        scores = [vqa_score(r["output"], r["answers"]) for r in records]
        return {"accuracy": 100 * sum(scores) / max(len(scores), 1), "num_samples": float(len(records))}
