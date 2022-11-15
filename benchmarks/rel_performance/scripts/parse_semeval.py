import typer
from pathlib import Path

from spacy.lang.en import English
from spacy.tokens import Span


def main(data_file: Path):
    nlp = English()

    with data_file.open("r", encoding="utf-8") as f:
        for count, line in enumerate(f):
            if count % 4 == 0:
                splits = line.split("\t")
                assert len(splits) == 2
                nr: int = int(splits[0].strip())
                sent: str = splits[1].strip()
                assert len(sent) > 2
                assert sent.startswith('"')
                assert sent.endswith('"')
                orig_sent = sent[1:-2]
                clean_sent, e1_s, e1_e, e2_s, e2_e = _parse_inline_entities(orig_sent)
                doc = nlp(clean_sent)
                e1: Span = doc.char_span(e1_s, e1_e, label="Entity")
                e2 = doc.char_span(e2_s, e2_e, label="Entity")
                if e1 is None or e2 is None:
                    # TODO: 5 sentences out of 8000 have punctuation typos
                    pass
                    # print(f"Couldn't parse {nr} {orig_sent}")
                else:
                    doc.set_ents([e1, e2], default="outside")
                    # print()
                    # print(orig_sent)
                    print(nr, doc.text, doc.ents)


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
