from pathlib import Path

import spacy
import typer
from spacy.tokens.span import Span
from spacy.tokens import DocBin

DEFAULT_INCORRECT_KEY = "incorrect_spans"


def main(pretrained_model, input_text_loc: Path, corpus_dir: Path, annotate_style: str):
    if annotate_style == "SILVER":
        silver(pretrained_model, input_text_loc, corpus_dir / "silver.spacy")
    elif annotate_style == "RELABEL":
        relabel(pretrained_model, input_text_loc, corpus_dir / "relabeled.spacy")
    elif annotate_style == "INCORRECT":
        mark_incorrect(pretrained_model, input_text_loc, corpus_dir / "incorrect.spacy", keep_ents=True)
    elif annotate_style == "INCORRECT_ONLY":
        mark_incorrect(pretrained_model, input_text_loc, corpus_dir / "incorrect_only.spacy", keep_ents=False)
    elif annotate_style == "FILTER":
        remove_incorrect(pretrained_model, input_text_loc, corpus_dir / "filtered.spacy")


def silver(pretrained_model, input_text_loc: Path, output_file: Path):
    nlp = spacy.load(pretrained_model)
    doc_bin = DocBin()
    with input_text_loc.open() as f:
        lines = f.readlines()
        for line in lines:
            doc = nlp(line)
            doc_bin.add(doc)
    doc_bin.to_disk(output_file)


def relabel(pretrained_model, input_text_loc: Path, output_file: Path):
    nlp = spacy.load(pretrained_model)
    doc_bin = DocBin()
    with input_text_loc.open() as f:
        lines = f.readlines()
        for line in lines:
            doc = nlp(line)
            new_ents = []
            for ent in doc.ents:
                if ent.text == "Emerson":
                    # Forcing the annotation to read "PERSON" for "Emerson"
                    span = Span(doc, ent.start, ent.end, "PERSON")
                    new_ents.append(span)
                else:
                    new_ents.append(ent)
            doc.ents = new_ents
            doc_bin.add(doc)

    doc_bin.to_disk(output_file)


def mark_incorrect(pretrained_model, input_text_loc: Path, output_file: Path, keep_ents: bool):
    nlp = spacy.load(pretrained_model)
    ner = nlp.get_pipe("beam_ner")
    incorrect_key = ner.incorrect_spans_key
    if incorrect_key is None:
        incorrect_key = DEFAULT_INCORRECT_KEY
    doc_bin = DocBin()
    with input_text_loc.open() as f:
        lines = f.readlines()
        for line in lines:
            doc = nlp(line)
            new_ents = []
            incorrect_spans = []
            for ent in doc.ents:
                if ent.text == "Emerson":
                    incorrect_spans.append(ent)
                else:
                    new_ents.append(ent)
            doc.spans[incorrect_key] = incorrect_spans
            if keep_ents:
                doc.ents = new_ents
            else:
                doc.ents = []
            doc_bin.add(doc)

    doc_bin.to_disk(output_file)


def remove_incorrect(pretrained_model, input_text_loc: Path, output_file: Path):
    nlp = spacy.load(pretrained_model)
    doc_bin = DocBin()
    with input_text_loc.open() as f:
        lines = f.readlines()
        for line in lines:
            doc = nlp(line)
            new_ents = []
            for ent in doc.ents:
                if ent.text != "Emerson":
                    new_ents.append(ent)
            doc.ents = new_ents
            doc_bin.add(doc)

    doc_bin.to_disk(output_file)


if __name__ == "__main__":
    typer.run(main)