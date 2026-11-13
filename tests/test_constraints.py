"""The option automaton and the compiled token masks."""

from __future__ import annotations

import random

import pytest

from option_grpo.options.constraints import ConstraintEngine, ConstraintSpec, LanguageTables, OptionAutomaton, VocabIndex
from option_grpo.options.option import CERTIFIED


@pytest.fixture()
def lang(index):
    return LanguageTables.from_index(index)


def _option(vocab, names):
    return next(o for o in vocab.options if o.call_names == tuple(names))


def _run(vocab, lang, names, text):
    automaton = OptionAutomaton(ConstraintSpec.from_option(_option(vocab, names), lang))
    ok = automaton.feed(text)
    return ok, automaton.terminated, automaton.progress


PATH = ("read_csv", "where", "collect")


@pytest.mark.parametrize(
    "text",
    [
        'df = tinyframe.read_csv(path)\nresult = df.where(tinyframe.col("age") > 30).collect()',
        'result = (\n    tinyframe.read_csv(path)\n    .where(tinyframe.col("a.b(").gt(3))\n    .collect()\n)',
        "result = tinyframe.read_csv(str(path)).where(tinyframe.col('a').between(1, 2.5)).collect()",
        '# df.select(\nx = "df.select("\nresult = tinyframe.read_csv(path).where(tinyframe.lit(1)).collect()',
    ],
)
def test_automaton_accepts_programs_that_follow_the_path(vocab, lang, text):
    assert _run(vocab, lang, PATH, text) == (True, True, 3)


@pytest.mark.parametrize(
    "text, reason",
    [
        ('df = tinyframe.read_csv(p).select("a")', "call outside the path on the chain"),
        ("df = tinyframe.read_csv(p).collect()", "path functions out of order"),
        ('df = tinyframe.read_csv(tinyframe.connect("x"))', "non-helper SDK call in arguments"),
        ("df = tinyframe. read_csv(p)", "whitespace after the dot"),
        ("df = tinyframe.read_csv + 1", "path function referenced without being called"),
        ("df = tinyframe.read_csv(p).where(tinyframe.col('a')).collect", "last call never made"),
        ("df = tinyframe.frame(p)", "a module name called like a function"),
    ],
)
def test_automaton_rejects_violations(vocab, lang, text, reason):
    ok, terminated, _ = _run(vocab, lang, PATH, text + "\n")
    assert not (ok and terminated), reason


def test_automaton_terminates_exactly_when_last_call_closes(vocab, lang):
    automaton = OptionAutomaton(ConstraintSpec.from_option(_option(vocab, ["where"]), lang))
    assert automaton.feed("result = frame.where(tinyframe.col('age').gt(1)")
    assert not automaton.terminated and automaton.progress == 1
    assert automaton.feed(")")
    assert automaton.terminated
    # Text after termination belongs to the high-level policy and is never checked.
    assert automaton.feed(".select('x')")


def test_pending_dot_makes_first_name_an_attribute(vocab, lang, tokenizer):
    engine = ConstraintEngine(VocabIndex.from_tokenizer(tokenizer), lang, vocab.options)
    where = _option(vocab, ["where"])
    automaton = engine.start(where.index, "result = frame.")
    assert automaton.pdot
    assert automaton.feed("where(") and automaton.progress == 1
    automaton = engine.start(where.index, "result = frame.")
    assert not automaton.feed("select(")


def test_vocab_index_recovers_exact_token_text(tokenizer):
    index = VocabIndex.from_tokenizer(tokenizer)
    text = 'result = tinyframe.read_csv(path).where(tinyframe.col("age") > 30)\n'
    ids = tokenizer(text)["input_ids"]
    assert "".join(index.strings[i] for i in ids) == text
    assert tokenizer.eos_token_id in index.always_banned


def test_masks_agree_with_exhaustive_simulation(vocab, lang, tokenizer, tasks):
    """The compiled mask must equal token-by-token simulation for every vocabulary entry.

    States are collected by replaying reference programs through options and by random
    walks over admissible tokens, which covers attribute, argument, string and comment
    contexts.
    """
    for option in vocab.options:
        option.status = CERTIFIED
    vocab_index = VocabIndex.from_tokenizer(tokenizer)
    engine = ConstraintEngine(vocab_index, lang, vocab.options)
    rng = random.Random(0)
    banned_always = set(vocab_index.always_banned)
    checked = 0

    def check(automaton):
        nonlocal checked
        spec = engine.mask(automaton)
        allowed = set(spec.ids.tolist()) if spec.mode == "allow" else set(range(vocab_index.size)) - set(spec.ids.tolist())
        expected = {
            tid for tid, text in enumerate(vocab_index.strings)
            if tid not in banned_always and automaton.copy().step(text)
        }
        assert allowed == expected, (automaton.key(), sorted(allowed ^ expected)[:10])
        checked += 1
        return sorted(allowed)

    # Replay reference programs token by token under the atomic option of their first call.
    by_function = vocab.by_function(0)
    for task in tasks.select(level=1)[:25]:
        option = by_function[task.path[0]]
        automaton = engine.start(option.index, "")
        for token in tokenizer(task.reference)["input_ids"]:
            if automaton.terminated:
                break
            allowed = check(automaton)
            if token not in allowed:
                break
            engine.advance(automaton, token)
    # Random walks over admissible tokens for a few composed options.
    for option in rng.sample(vocab.level(1), 8):
        for _ in range(3):
            automaton = engine.start(option.index, "")
            for _ in range(40):
                if automaton.terminated:
                    break
                allowed = check(automaton)
                if not allowed:
                    break
                engine.advance(automaton, rng.choice(allowed))
                assert not automaton.violated
    assert checked > 300
    info = engine.cache_info()
    assert info["hits"] > 0 and info["entries"] == info["misses"]
