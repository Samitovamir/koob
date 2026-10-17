"""`cb memory eval` and `cb memory eval-retrieval` (spec 19).

The live brain is never imported here (this package must not import `koob`): `--live`
needs `--brain module:factory`, a callable returning a `runner.Brain`. Pytest never passes it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(help="Память Куба: оценка (cb memory eval) и тест поиска (eval-retrieval).")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_set(spec: str) -> tuple[Path, bool]:
    """(root, hidden). `dev` is the tracked set; anything else is a path given at run time."""
    if spec == "dev":
        return _repo_root() / "evals" / "memory" / "dev", False
    p = Path(spec).expanduser()
    return p, True  # any set other than the tracked dev one is graded as hidden: scores only


@app.command("eval")
def eval_cmd(
    set_: Annotated[
        str,
        typer.Option("--set", help="dev, либо путь к набору (скрытый — только оценки по группам)."),
    ] = "dev",
    group: Annotated[
        list[str] | None, typer.Option("--group", "-g", help="Только эти группы.")
    ] = None,
    scenario: Annotated[
        list[str] | None, typer.Option("--scenario", help="Только эти id (не для скрытого набора).")
    ] = None,
    live: Annotated[
        bool, typer.Option("--live", help="Настоящий мозг (нужен --brain). Не для pytest.")
    ] = False,
    brain: Annotated[str | None, typer.Option("--brain", help="module:factory для --live.")] = None,
    json_out: Annotated[
        str | None, typer.Option("--json", help="Куда записать JSON-отчёт.")
    ] = None,
    scale: Annotated[str, typer.Option("--scale", help="1x | 10x | 100x — множитель шума.")] = "1x",
    canary: Annotated[
        bool, typer.Option("--canary", help="Только сценарии с канарейками (§19.5).")
    ] = False,
    judge: Annotated[
        str | None,
        typer.Option("--judge", help="ollama:<model> — локальный судья для свободного текста."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    sut: Annotated[
        str,
        typer.Option(
            "--sut", help="api (по умолчанию) | plain | legacy — базовые линии без модели."
        ),
    ] = "api",
    per_group: Annotated[
        int | None, typer.Option("--per-group", help="plain/legacy: N сценариев на группу (seed).")
    ] = None,
    tag: Annotated[
        str, typer.Option("--tag", help="plain/legacy: имя подмножества в отчёте (B, C…).")
    ] = "",
    legacy_cache: Annotated[
        bool, typer.Option("--legacy-cache", help="legacy: разбор YAML заметок один раз (быстрее).")
    ] = False,
    extract: Annotated[
        str,
        typer.Option(
            "--extract",
            help="off (по умолчанию, бесплатно) | cloud | local — разбор чатов моделью на "
            "точках расписания; ответы кэшируются на диске, повторный прогон без вызовов.",
        ),
    ] = "off",
    labeler: Annotated[
        str, typer.Option("--labeler", help="auto | on | off — метки безопасности (ждёт прогрева).")
    ] = "auto",
    extract_model: Annotated[
        str | None,
        typer.Option(
            "--extract-model",
            help="модель облачного разбора для этого прогона (по умолчанию из настроек); "
            "кэш хранится отдельно по модели.",
        ),
    ] = None,
) -> None:
    """Run the memory scenarios and print a table by group."""
    from .judge import NoJudge, OllamaJudge
    from .loader import ScenarioError, load_set
    from .runner import run_set

    root, hidden = _resolve_set(set_)
    try:
        base = None
        if hidden:  # a hidden scenario may use a dev persona
            dev_p = _repo_root() / "evals" / "memory" / "dev" / "personas.yaml"
            base = yaml.safe_load(dev_p.read_text(encoding="utf-8")) if dev_p.exists() else None
        loaded = load_set(
            root, redact=hidden, expect_set=None if hidden else "dev", base_personas=base
        )
    except ScenarioError as exc:
        typer.echo(f"Набор не прошёл проверку формата: {exc}", err=True)
        raise typer.Exit(2) from exc
    if hidden:
        for scn in loaded.scenarios:
            scn.set = "hidden"
    if hidden and scenario:
        typer.echo("--scenario не работает со скрытым набором.", err=True)
        raise typer.Exit(2)
    if extract not in ("off", "cloud", "local") or labeler not in ("auto", "on", "off"):
        typer.echo("--extract: off | cloud | local; --labeler: auto | on | off", err=True)
        raise typer.Exit(2)
    if sut != "api":
        _bench(loaded, sut, group, scenario, scale, seed, json_out, per_group, legacy_cache, tag)
        return
    brain_obj = None
    if live:
        if not brain:
            typer.echo(
                "--live нужен --brain module:factory (мозг подключает слой runtime; в harness его нет).",
                err=True,
            )
            raise typer.Exit(2)
        mod, _, fn = brain.partition(":")
        brain_obj = getattr(importlib.import_module(mod), fn)()
    judge_obj = NoJudge()
    if judge:
        kind, _, model = judge.partition(":")
        if kind != "ollama":
            typer.echo("--judge: поддержан только ollama:<model>", err=True)
            raise typer.Exit(2)
        judge_obj = OllamaJudge(model or "qwen3:14b")
    try:
        from .sut import ApiSUT

        lab: bool | str = {"auto": "auto", "on": True, "off": False}[labeler]
        report = run_set(
            loaded, groups=group, scenario_ids=scenario, only_canary=canary, brain=brain_obj, judge=judge_obj,
            scale=scale, seed=seed, sut_factory=lambda: ApiSUT(extract=extract, labeler=lab, extract_model=extract_model),
        )  # fmt: skip
        report.meta["extract"] = extract
        if extract_model:
            report.meta["extract_model"] = extract_model
    finally:
        judge_obj.close()
    typer.echo(report.table(verbose=verbose and not hidden))
    if json_out:
        report.write_json(json_out)
        typer.echo(f"JSON: {json_out}")
    if os.environ.get("CB_MEMORY_EVAL_EXIT_CODE") == "1" and not all(
        report.gates("hidden" if hidden else "dev").values()
    ):
        raise typer.Exit(1)


def _bench(
    loaded,
    sut: str,
    group,
    scenario,
    scale: str,
    seed,
    json_out,
    per_group,
    legacy_cache=False,
    tag="",
) -> None:  # noqa: ANN001
    """Memory-level baselines (plain file / legacy memory): no model, no brain."""
    import time

    from . import bench

    if sut not in ("plain", "legacy"):
        typer.echo("--sut: api | plain | legacy", err=True)
        raise typer.Exit(2)
    factory = (
        bench.PlainFileSUT if sut == "plain" else (lambda: bench.LegacySUT(cache=legacy_cache))
    )
    seed = bench.DEFAULT_SEED if seed is None else seed
    t0 = time.perf_counter()
    res = bench.run_bench(
        loaded,
        factory,
        groups=group,
        scenario_ids=scenario,
        scale=scale,
        seed=seed,
        per_group=per_group,
    )
    summary = bench.summarize(res["records"])
    typer.echo(
        f"sut={sut} scale={scale} seed={seed} scenarios={res['scenarios']} admitted={res['admitted']}"
    )
    for g, m in summary.items():
        typer.echo(
            f"{g:14} hit={m['hit']} stale={m['stale']} tokens={m['tokens_mean']} "
            f"p50={m['ms_p50']}ms p95={m['ms_p95']}ms na={m['na']}"
        )
    for e in res["errors"]:
        typer.echo(f"error: {e}", err=True)
    if json_out:
        out = {
            "sut": sut + ("+cache" if sut == "legacy" and legacy_cache else ""),
            "scale": scale,
            "seed": seed,
            "subset": tag or ("full" if not (group or per_group or scenario) else "B"),
            "groups": group,
            "per_group": per_group,
            "legacy_cache": legacy_cache,
            "wall_s": round(time.perf_counter() - t0, 1),
        }
        out |= {k: res[k] for k in ("scenarios", "admitted", "misses", "errors")}
        out["summary"] = summary
        bench.write_json(json_out, out)
        typer.echo(f"JSON: {json_out}")


def _bench_doc(sut: str, scale: str, seed: int, res: dict, wall: float, **extra) -> dict:  # noqa: ANN003
    from . import bench

    out = {
        "sut": sut,
        "scale": scale,
        "seed": seed,
        "subset": extra.get("tag") or "full",
        "groups": extra.get("groups"),
        "per_group": extra.get("per_group"),
        "legacy_cache": extra.get("legacy_cache", False),
        "wall_s": round(wall, 1),
    }
    out |= {k: res[k] for k in ("scenarios", "admitted", "misses", "errors")}
    out["labeler"] = res.get("labeler", "n/a")
    out["summary"] = bench.summarize(res["records"])
    return out


@app.command("bench-report")
def bench_report_cmd(
    inputs: Annotated[
        list[str] | None, typer.Argument(help="JSON прогонов `eval --sut ... --json`.")
    ] = None,
    out_json: Annotated[str, typer.Option("--out-json")] = "baseline.json",
    out_md: Annotated[str, typer.Option("--out-md")] = "baseline.md",
    date: Annotated[str, typer.Option("--date")] = "",
    method: Annotated[str, typer.Option("--method-file", help="Файл с текстом метода.")] = "",
    sut: Annotated[
        str | None,
        typer.Option("--sut", help="api: прогнать настоящий движок здесь же (без модели)."),
    ] = None,
    scale: Annotated[
        list[str] | None, typer.Option("--scale", help="для --sut api; по умолчанию 1x.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    per_group: Annotated[int | None, typer.Option("--per-group")] = None,
    extract: Annotated[
        str,
        typer.Option(
            "--extract",
            help="для --sut api: off | cloud | local — разбор чатов моделью (кэш на диске общий с eval).",
        ),
    ] = "off",
) -> None:
    """Collect runs into baseline.json (numbers only) and baseline.md.

    `--sut api` runs the dev set itself through ApiBenchSUT (what the brain sees: core + recall +
    search) at each --scale and adds columns to the runs loaded from files."""
    import json
    import time

    from . import bench
    from .loader import load_set

    keys = ("summary", "misses", "errors", "scenarios", "admitted", "wall_s", "seed")
    runs: dict = {}
    commands: list[str] = []
    docs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in inputs or []]
    if sut is not None:
        if sut != "api":
            typer.echo("--sut: только api", err=True)
            raise typer.Exit(2)
        loaded = load_set(_repo_root() / "evals" / "memory" / "dev", redact=False, expect_set="dev")
        if extract not in ("off", "cloud", "local"):
            typer.echo("--extract: off | cloud | local", err=True)
            raise typer.Exit(2)
        the_seed = bench.DEFAULT_SEED if seed is None else seed
        for sc in scale or ["1x"]:
            t0 = time.perf_counter()
            res = bench.run_bench(
                loaded,
                lambda: bench.ApiBenchSUT(extract=extract),
                scale=sc,
                seed=the_seed,
                per_group=per_group,
            )
            docs.append(
                _bench_doc(
                    "api" if extract == "off" else f"api+{extract}",
                    sc,
                    the_seed,
                    res,
                    time.perf_counter() - t0,
                    per_group=per_group,
                    tag="B" if per_group else "",
                )
            )
    for d in docs:
        label = d["scale"] + ("" if d.get("subset") == "full" else " " + d["subset"])
        runs.setdefault(d["sut"], {})[label] = {k: d[k] for k in keys}
        flags = "".join(f" --group {g}" for g in d.get("groups") or [])
        flags += f" --per-group {d['per_group']}" if d.get("per_group") else ""
        flags += f" --tag {d['subset']}" if d.get("subset") not in (None, "full") else ""
        flags += " --legacy-cache" if d.get("legacy_cache") else ""
        name = d["sut"].split("+")[0]
        commands.append(
            f"uv run cb memory eval --sut {name} --scale {d['scale']}{flags} --json run.json"
            if name != "api" or not sut
            else f"uv run cb memory bench-report --sut api --scale {d['scale']}{flags}"
            + (f" --extract {d['sut'].split('+')[1]}" if "+" in d["sut"] else "")
        )
    meta = {
        "date": date,
        "set": "dev",
        "token_counter": bench.TOKEN_COUNTER,
        "method": Path(method).read_text(encoding="utf-8") if method else "",
        "commands": [
            "# env: KOOB_DATA_DIR=$PWD/.devdata/data KOOB_WORKSPACE_DIR=$PWD/.devdata/workspace",
            *commands,
            "uv run cb memory bench-report <the run.json files> --out-json b.json --out-md b.md",
        ],
    }
    numbers = {k: v for k, v in meta.items() if k != "method"}
    bench.write_json(out_json, {"meta": numbers, "runs": runs})
    Path(out_md).write_text(bench.render_markdown({"meta": meta, "runs": runs}), encoding="utf-8")
    typer.echo(f"{out_json}, {out_md}")


@app.command("eval-extract")
def eval_extract_cmd(
    cases: Annotated[Path, typer.Option("--set", help="JSONL: {id, raws, relations?, gold}")],
    processor: Annotated[str, typer.Option(help="cloud | local")] = "local",
    out: Annotated[Path | None, typer.Option(help="per-case numbers as JSON")] = None,
) -> None:
    """Extraction quality against gold (§6.3): the user's sample, read by `cb` only; prints
    numbers, no message text."""
    import json

    from ..workers import extract

    rows = [json.loads(ln) for ln in cases.read_text(encoding="utf-8").splitlines() if ln.strip()]
    proc = extract.CloudExtractor() if processor == "cloud" else extract.OllamaExtractor()
    report = extract.evaluate(rows, proc)
    if out is not None:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(json.dumps({"processor": proc.model_id, **report["total"]}))


@app.command("eval-retrieval")
def eval_retrieval_cmd(
    set_: Annotated[str, typer.Option("--set")] = "evals/memory/dev/crosslingual/set.yaml",
    embedder: Annotated[
        list[str] | None,
        typer.Option("--embedder", help="embeddinggemma-300m | multilingual-e5-small | ..."),
    ] = None,
    mode: Annotated[
        str, typer.Option("--mode", help="embedder | full (эмбеддер + алиасы)")
    ] = "embedder",
    json_out: Annotated[str | None, typer.Option("--json")] = None,
    cache_dir: Annotated[
        str | None, typer.Option("--cache-dir", help="Каталог моделей fastembed.")
    ] = None,
) -> None:
    """recall@1, recall@3 and query latency for fastembed embedders on the cross-language set."""
    import json
    from dataclasses import asdict

    from . import retrieval

    names = embedder or ["embeddinggemma-300m", "multilingual-e5-small"]
    path = Path(set_)
    if not path.is_absolute() and not path.exists():
        path = _repo_root() / set_
    data = retrieval.load_set(path)
    out = []
    for name in names:
        spec = retrieval.EMBEDDERS.get(name)
        if spec is None:
            typer.echo(
                f"Неизвестный эмбеддер {name}; есть: {', '.join(retrieval.EMBEDDERS)}", err=True
            )
            raise typer.Exit(2)
        emb = retrieval.FastEmbedder(spec, cache_dir)
        res = retrieval.run_bench(data, emb, mode=mode, name=name)
        typer.echo(retrieval.format_result(res))
        out.append(asdict(res))
    if json_out:
        Path(json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
