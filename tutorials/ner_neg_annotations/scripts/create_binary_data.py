from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
from spacy.tokens.doc import SetEntsDefault
from spacy.training import Corpus, Example

DEFAULT_INCORRECT_KEY = "incorrect_spans"


def main(
    pretrained_model,
    corpus_loc: Path,
    output_dir: Path,
):
    corpus = Corpus(corpus_loc)
    nlp = spacy.load(pretrained_model)
    ner = nlp.get_pipe("beam_ner")
    incorrect_key = ner.incorrect_spans_key or DEFAULT_INCORRECT_KEY
    doc_bin = DocBin()
    for example in parse(nlp, corpus(nlp)):
        correct_preds, incorrect_preds, missing_preds = evaluate_preds(example)
        if missing_preds or incorrect_preds:
            # If there are mistakes, we use 'missing', not 'O'
            example.x.set_ents(
                correct_preds,
                default=SetEntsDefault.missing
            )
            example.x.spans[incorrect_key] = incorrect_preds
        else:
            # If we have it entirely correct, let it learn 'O'.
            example.x.set_ents(
                correct_preds,
                default=SetEntsDefault.outside
            )
        doc_bin.add(example.x)
    doc_bin.to_disk(output_dir / f"{corpus_loc.name}_correct_incorrect.spacy")


def parse(nlp, examples):
    texts_and_examples = (
        (eg.reference.text, eg) for eg in examples
        if all(w.ent_iob != 0 for w in eg.y)
    )
    for doc, example in nlp.pipe(texts_and_examples, as_tuples=True):
        yield Example(doc, example.reference)


def evaluate_preds(example):
    correct_preds = []
    incorrect_preds = []
    missing_preds = []
    golds = {
        (e.label_, e.start, e.end): e
        for e in example.get_aligned_spans_y2x(example.y.ents)
    }
    for pred_span in example.x.ents:
        key = (pred_span.label_, pred_span.start, pred_span.end)
        if key in golds:
            correct_preds.append(pred_span)
            golds.pop(key)
        else:
            incorrect_preds.append(pred_span)
    return correct_preds, incorrect_preds, golds.values()


if __name__ == "__main__":
    typer.run(main)
