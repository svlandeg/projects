from pathlib import Path
import random

import spacy
import typer
from spacy.tokens import DocBin, Doc
from spacy.tokens.doc import SetEntsDefault
from spacy.training import Corpus

DEFAULT_INCORRECT_KEY = "incorrect_spans"


def main(
    pretrained_model,
    corpus_loc: Path, output_dir: Path,
    keep_correct: bool,
    keep_incorrect: bool,
    keep_missing: bool,
    teach_prob: float=1.0,
    o_prob: float=0.05
):
    corpus = Corpus(corpus_loc)

    nlp = spacy.load(pretrained_model)
    ner = nlp.get_pipe("beam_ner")
    incorrect_key = ner.incorrect_spans_key
    if incorrect_key is None:
        incorrect_key = DEFAULT_INCORRECT_KEY
    doc_bin = DocBin()
    for example in parse(nlp, corpus(nlp)):
        if random.random() >= teach_prob:
            doc_bin.add(example.reference)
        else:
            correct_preds, incorrect_preds, missing_preds = evaluate_preds(example)
            new_ents = []
            if keep_correct:
                new_ents.extend(correct_preds)
            if keep_missing:
                new_ents.extend(missing_preds)
            doc_clean = Doc(
                nlp.vocab,
                words=[w.text for w in example.y],
                spaces=[bool(w.whitespace_) for w in example.y]
            )
            if new_ents:
                outsides = []
                for token in example.reference:
                    if token.ent_iob_ == "O" and random.random() < o_prob:
                        outsides.append(doc_clean[token.i:token.i+1])
                doc_clean.set_ents(
                    new_ents,
                    outside=outsides,
                    default=SetEntsDefault.missing
                )
            else:
                doc_clean.set_ents([], default=SetEntsDefault.outside)
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


def parse(nlp, examples):
    texts_and_examples = ((eg.reference.text, eg) for eg in examples)
    for doc, example in nlp.pipe(texts_and_examples, as_tuples=True):
        example.predicted = doc
        yield example


def evaluate_preds(example):
    correct_preds = []
    incorrect_preds = []
    missing_preds = []
    if not example.y.has_annotation("ENT_IOB"):
        return correct_preds, incorrect_preds, missing_preds

    gold_spans = example.y.ents
    example.get_aligned_spans_y2x(example.y.ents)
    gold_tuples = {(e.label_, e.start, e.end) for e in gold_spans}

    for pred_span in example.get_aligned_spans_x2y(example.x.ents):
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
