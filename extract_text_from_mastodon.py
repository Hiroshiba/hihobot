import argparse
import html
import json
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any


def strip_p_tag(s: str, _re=re.compile(r"<p>(.*)</p>")):
    """
    >>> strip_p_tag("<p>  </p>  </p>")
    '  </p>  '
    """
    return _re.sub("\\1", s)


def replace_new_line(s: str, _re=re.compile(r"<br.*?/?>")):
    """
    >>> replace_new_line("test<br>test")
    'test\\ntest'
    >>> replace_new_line("test<br />test")
    'test\\ntest'
    """
    return _re.sub("\n", s)


def eliminate_tag(s: str, _re=re.compile(r'<.*?>')):
    """
    >>> eliminate_tag("<a link=hoge> test </a>")
    ' test '
    """
    return _re.sub("", s)


def unescape_html(s: str):
    """
    >>> unescape_html("&lt")
    '<'
    """
    return html.unescape(s)


def eliminate_username(s: str, _re=re.compile(r":?@\w+:?")):
    """
    >>> eliminate_username("hogehoge :@hogehoge: hogehoge")
    'hogehoge  hogehoge'
    """
    return _re.sub("", s)


def eliminate_nico_id(s: str, _re=re.compile(r"(lv\d+)|(sm\d+)")):
    """
    >>> eliminate_nico_id("sm12345 lv45678")
    ' '
    """
    return _re.sub("", s)


def eliminate_hash_tag(s: str, _re=re.compile(r"#[^\s]+")):
    """
    >>> eliminate_hash_tag("test1 #test2 test3")
    'test1  test3'
    """
    return _re.sub("", s)


def has_uri(s: str, _re=re.compile(r"(https?|ftp|file)(://[-_.!~*\'()a-zA-Z0-9;/?:@&=+$,%#]+)")):
    """
    >>> has_uri("hoge://fuga.com")
    False
    >>> has_uri("https://qiita.com/?hoge=fuga")
    True
    """
    return _re.search(s) is not None


def has_enquete(s: str):
    """
    >>> has_enquete("-----friends.nico アンケート(結果)-----")
    True
    """
    return "friends.nico アンケート" in s


def is_only_number(s: str, _re=re.compile(r"^\d+$")):
    return _re.match(s) is not None


def is_only_dot(s: str, _re=re.compile(r"^\.+$")):
    return _re.match(s) is not None


def is_only_arrow(s: str, _re=re.compile(r"^[←↓↑→]$")):
    return _re.match(s) is not None


def contain_unknown_chars(s: str, chars: str):
    """
    >>> contain_unknown_chars("hogehoge", chars="hoge")
    False
    >>> contain_unknown_chars("hogehoge", chars="hog")
    True
    """
    return len(set(s) - set(chars)) > 0


def clean_up_text(obj: Dict[str, Any]):
    if "content" not in obj:  # ?
        return None

    if obj["summary"] is not None:  # CW
        return None

    s: str = obj["content"]
    s = strip_p_tag(s)
    s = replace_new_line(s)
    s = eliminate_tag(s)
    s = unescape_html(s)
    s = eliminate_username(s)
    s = eliminate_hash_tag(s)
    s = eliminate_nico_id(s)
    s = s.strip()

    if has_uri(s):
        return None

    if has_enquete(s):
        return None

    if is_only_number(s):
        return None

    if is_only_dot(s):
        return None

    if is_only_arrow(s):
        return None

    return s


def extract_text_from_mastodon(
        mastodon_outbox: Path,
        output_texts: Path,
):
    outbox = json.load(mastodon_outbox.open(encoding='UTF8'))
    objs = [item["object"] for item in outbox["orderedItems"]]
    texts = filter(None, Pool().map(clean_up_text, objs))
    output_texts.open('w', encoding='UTF8').writelines([text + "\n" for text in texts])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mastodon_outbox', type=Path)
    parser.add_argument('--output_texts', type=Path, default=Path('texts.txt'))
    args = parser.parse_args()

    extract_text_from_mastodon(
        mastodon_outbox=args.mastodon_outbox,
        output_texts=args.output_texts,
    )
