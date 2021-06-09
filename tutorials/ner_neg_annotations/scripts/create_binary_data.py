from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
from spacy.training import Corpus

DEFAULT_INCORRECT_KEY = "incorrect_spans"


def main(pretrained_model, corpus_loc: Path, output_dir: Path, keep_correct: bool, keep_incorrect: bool, keep_missing: bool):
    corpus = Corpus(corpus_loc)

    nlp = spacy.load(pretrained_model)
    ner = nlp.get_pipe("beam_ner")
    incorrect_key = ner.incorrect_spans_key
    if incorrect_key is None:
        incorrect_key = DEFAULT_INCORRECT_KEY
    doc_bin = DocBin()
    for example in corpus(nlp):
        text = example.reference.text
        example.predicted = nlp(text)
        doc_clean = nlp.make_doc(text)
        correct_preds, incorrect_preds, missing_preds = evaluate_preds(example)

        new_ents = []
        if keep_correct:
            new_ents.extend(correct_preds)
        if keep_missing:
            new_ents.extend(missing_preds)
        if new_ents:
            doc_clean.ents = new_ents
        if keep_incorrect:
            doc_clean.spans[incorrect_key] = incorrect_preds
        if doc_clean.ents or doc_clean.spans.get(incorrect_key):
            doc_bin.add(doc_clean)

    file_name = corpus_loc.name
    if keep_correct:
        file_name += "_correct"
    if keep_incorrect:
        file_name += "_incorrect"
    if keep_missing:
        file_name += "_missing"
    file_name += ".spacy"
    output_file = output_dir / file_name
    doc_bin.to_disk(output_file)


def evaluate_preds(example):
    correct_preds = []
    incorrect_preds = []
    missing_preds = []
    if not example.y.has_annotation("ENT_IOB"):
        return correct_preds, incorrect_preds, missing_preds

    gold_spans = example.get_aligned_spans_y2x(example.y.ents)
    gold_tuples = {(e.label_, e.start, e.end) for e in gold_spans}

    for pred_span in example.x.ents:
        pred_tuple = (pred_span.label_, pred_span.start, pred_span.end)
        if pred_tuple in gold_tuples:
            correct_preds.append(pred_span)
            gold_tuples.remove(pred_tuple)
        else:
            incorrect_preds.append(pred_span)
    for gold_span in gold_spans:
        gold_tuple = (gold_span.label_, gold_span.start, gold_span.end)
        if gold_tuple in gold_tuples:
            missing_preds.append(gold_span)
    return correct_preds, incorrect_preds, missing_preds


if __name__ == "__main__":
    typer.run(main)
