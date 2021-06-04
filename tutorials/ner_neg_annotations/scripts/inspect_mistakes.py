import spacy
import typer
from spacy.tokens import DocBin


def main(pretrained_model, gold_file):
    nlp = spacy.load(pretrained_model)
    doc_bin = DocBin().from_disk(gold_file)
    for gold_doc in doc_bin.get_docs(nlp.vocab):
        text = gold_doc.text
        print(text)
        pred_doc = nlp(text)
        gold_ents = set()
        gold_ents_no_label = set()
        for ent in gold_doc.ents:
            gold_ents.add((ent.start, ent.end, ent.text, ent.label_))
            gold_ents_no_label.add((ent.start, ent.end, ent.text))
        pred_ents = set()
        pred_ents_no_label = set()
        for ent in pred_doc.ents:
            pred_ents.add((ent.start, ent.end, ent.text, ent.label_))
            pred_ents_no_label.add((ent.start, ent.end, ent.text))
        for (start, end, text, label_) in gold_ents:
            if (start, end, text, label_) not in pred_ents:
                if (start, end, text) not in pred_ents_no_label:
                    print("  Missing prediction:", start, end, text, label_)
        for (start, end, text, label_) in pred_ents:
            if (start, end, text, label_) not in gold_ents:
                if (start, end, text) not in gold_ents_no_label:
                    print("  Wrongly predicted:", start, end, text, label_)
        for (start, end, text, label_) in gold_ents:
            if (start, end, text, label_) not in pred_ents:
                if (start, end, text) in pred_ents_no_label:
                    print("  Label not correct:", start, end, text, label_)
        print()


if __name__ == "__main__":
    typer.run(main)