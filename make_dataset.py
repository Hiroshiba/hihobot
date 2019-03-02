import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import ndjson


def contain_unknown_chars(s: str, chars: str):
    """
    >>> contain_unknown_chars("hogehoge", chars="hoge")
    False
    >>> contain_unknown_chars("hogehoge", chars="hog")
    True
    """
    return len(set(s) - set(chars)) > 0


def make_dataset(
        texts_path: Path,
        num_chars: int,
        eliminate_words: List[str],
        out_text: Path,
        out_char: Path,
):
    texts: List[str] = texts_path.open(encoding='UTF8').read().split()
    texts = list(filter(lambda s: all(w not in s for w in eliminate_words), texts))

    counter = Counter("".join(texts))
    chars = "".join(c[0] for c in counter.most_common(num_chars))

    texts = list(filter(lambda s: not contain_unknown_chars(s, chars=chars), texts))

    ndjson.dump([{"str": s} for s in texts], out_text.open('w', encoding='UTF8'), ensure_ascii=False)
    json.dump([c for c in chars], out_char.open('w', encoding='UTF8'), ensure_ascii=False)

    # show alphabet
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c in chars:
            print(c, chars.index(c))
        else:
            print(c, "not exist")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--texts_path', type=Path, default=Path('texts.txt'))
    parser.add_argument('--num_chars', type=int, default=2048)
    parser.add_argument('--eliminate_words', type=str, nargs='*', default=[])
    parser.add_argument('--output_dataset_text', type=Path, default=Path('dataset_text.ndjson'))
    parser.add_argument('--output_dataset_char', type=Path, default=Path('dataset_char.json'))
    args = parser.parse_args()

    make_dataset(
        texts_path=args.texts_path,
        num_chars=args.num_chars,
        eliminate_words=args.eliminate_words,
        out_text=args.output_dataset_text,
        out_char=args.output_dataset_char,
    )
