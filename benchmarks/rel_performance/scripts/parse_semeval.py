import typer
from pathlib import Path

from spacy.lang.en import English
from spacy.tokens import Span, Doc, DocBin

from wasabi import msg

def main(data_input_file: Path, spacy_output_file: Path):
    nlp = English()
    docs = []
    rel_labels = set()
    Doc.set_extension("rel", default={})

    with data_input_file.open("r", encoding="utf-8") as f:
        for count, line in enumerate(f):
            # first line: the sentence with inline markup
            if count % 4 == 0:
                splits = line.split("\t")
                assert len(splits) == 2
                sent: str = splits[1].strip()
                assert len(sent) > 2
                assert sent.startswith('"')
                assert sent.endswith('"')
                orig_sent = sent[1:-2]
                clean_sent, e1_s, e1_e, e2_s, e2_e = _parse_inline_entities(orig_sent)
                doc = nlp(clean_sent)
                e1: Span = doc.char_span(e1_s, e1_e, label="Entity")
                e2: Span = doc.char_span(e2_s, e2_e, label="Entity")
                if e1 is None or e2 is None:
                    # TODO: 5 sentences out of 8000 have punctuation typos
                    doc = None
                    # print(f"Couldn't parse {nr} {orig_sent}")
                else:
                    doc.set_ents([e1, e2], default="outside")
                    rels = {}
                    rels[(e1.start, e2.start)] = {}
                    rels[(e2.start, e1.start)] = {}
                    # print()
                    # print(orig_sent)
                    # print(nr, doc.text, doc.ents)

            # second line: the label and direction of the relation
            if count % 4 == 1 and doc is not None:
                label: str = line.strip()
                if label.endswith("(e1,e2)"):
                    label = label.replace("(e1,e2)", "")
                    rels[(e1.start, e2.start)][label] = 1.0
                    rel_labels.add(label)
                elif label.endswith("(e2,e1)"):
                    label = label.replace("(e2,e1)", "")
                    rels[(e2.start, e1.start)][label] = 1.0
                    rel_labels.add(label)
                else:
                    assert label == "Other"
                    rels[(e1.start, e2.start)]["Other"] = 1.0
                    rels[(e2.start, e1.start)]["Other"] = 1.0
                    rel_labels.add("Other")
                doc._.rel = rels

            # line 3 (comment) is ignored

            # line 4 should be blank, ending this instance
            if count % 4 == 3 and doc is not None:
                assert line.strip() == ""
                docs.append(doc)

        # record all other relations as non-existing
        for doc in docs:
            ents = list(doc.ents)
            assert len(ents) == 2
            e1_i = ents[0].start
            e2_i = ents[1].start
            for label in rel_labels:
                if label not in doc._.rel[(e1_i, e2_i)]:
                    doc._.rel[(e1_i, e2_i)][label] = 0.0
                if label not in doc._.rel[(e2_i, e1_i)]:
                    doc._.rel[(e2_i, e1_i)][label] = 0.0

        doc_bin = DocBin(docs=docs, attrs=["ORTH", "ENT_IOB", "ENT_TYPE"], store_user_data=True)
        doc_bin.to_disk(spacy_output_file)

        msg.good(f"Written {len(docs)} to {spacy_output_file} containing REL labels {sorted(rel_labels)}")


def _parse_inline_entities(sentence: str):
    e1_start = sentence.index("<e1>")
    e1_end = sentence.index("</e1>")
    e2_start = sentence.index("<e2>")
    e2_end = sentence.index("</e2>")
    # assume the start is always before the end
    assert e1_start < e1_end
    assert e2_start < e2_end
    # assume e1 is entirely before e2
    assert e1_end < e2_start

    # adjust the offsets if the tags are removed (len 4 and 5)
    e1_end = e1_end - 4
    e2_start = e2_start - 9
    e2_end = e2_end - 13

    sentence = sentence.replace("<e1>", "")
    sentence = sentence.replace("</e1>", "")
    sentence = sentence.replace("<e2>", "")
    sentence = sentence.replace("</e2>", "")

    return sentence, e1_start, e1_end, e2_start, e2_end


if __name__ == "__main__":
    typer.run(main)
