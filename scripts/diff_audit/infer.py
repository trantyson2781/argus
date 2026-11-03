"""Strict check that every INFERENCE call claim (caller edges and call rows) cites a line with a real call to the claimed target.
Run: uv run python scripts/diff_audit/infer.py   (the repo paths and commit lists are hard-coded below; edit them)."""
import ast, random, sys
from pathlib import Path
from argus.explore.changes import build_diff_view
from argus.explore.sides import materialize_sides
from argus.extract import scan

def calls_on_line(root, file, line):
    tree = ast.parse((root/file).read_text()); names=set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and n.lineno <= line <= (n.end_lineno or n.lineno):
            f=n.func
            names.add(f.id if isinstance(f,ast.Name) else f.attr if isinstance(f,ast.Attribute) else None)
    return names

runs = [("/home/tran_thai_son_admin/Project/argus", ["be9fdd2","2c235d9","f7ee3a7","7868390","3ad8b6b"]),
        ("/home/tran_thai_son_admin/Project/stock_analysis", ["b37255c","c41b5fa","f3795d4","d3cefec","41cc9ce","40346d3"])]
total=ok=0; bad=[]; sample=[]
for repo, cs in runs:
    for c in cs:
        with materialize_sides(Path(repo), c+"~1", c) as (b,h):
            v = build_diff_view(b,h,scan(b.root),scan(h.root),True,True)
            items=[]
            for g in (*v.interface_callers,*v.affected_callers):
                for e in g.edges: items.append((e.callee, e.claim.evidence[0], b.root if e.was else h.root, f"{g.caller} -> {e.callee}"))
            for k in v.calls:
                items.append((k.target, k.claim.evidence[0], b.root if k.change=="removed" else h.root, f"{k.source} -> {k.target} ({k.change})"))
            for callee, ev, root, label in items:
                total+=1
                simple = callee.rsplit(".",1)[-1]
                names = calls_on_line(root, ev.file, ev.line)
                # constructor calls link to the class; __init__ callee is not expected here
                if simple in names: ok+=1
                else: bad.append((c,label,ev.location,sorted(n for n in names if n)))
                sample.append((c,label,ev.location))
print(f"INFERENCE call claims checked: {total}; a real call to that name on the cited line: {ok} ({100*ok/total:.1f}%)")
for x in bad[:10]: print("  NOT FOUND:", x)
random.seed(3)
print("\nrandom 30 for eyeballing (commit, claim, evidence):")
for s in random.sample(sample, min(30,len(sample))): print("  ", *s)
