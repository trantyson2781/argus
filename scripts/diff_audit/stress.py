"""Formatting-only and mutation stress test of `explore diff` on a throwaway clone.
Run: git clone <repo> /tmp/x && uv run python scripts/diff_audit/stress.py /tmp/x   (it commits to the clone, never to your repo)."""
import ast, random, subprocess, sys
from pathlib import Path
from argus.explore.changes import Flag, build_diff_view
from argus.explore.sides import materialize_sides
from argus.extract import scan

R = Path(sys.argv[1])
def git(*a): return subprocess.run(["git","-C",str(R),"-c","user.name=t","-c","user.email=t@t",*a],capture_output=True,check=True,text=True).stdout
def view():
    with materialize_sides(R, "HEAD~1", "HEAD") as (b, h):
        return build_diff_view(b,h,scan(b.root),scan(h.root),True,True)
files = [Path(f) for f in git("ls-files","*.py").split()]
# 1. formatting-only: ast.unparse everything
n=0
for f in files:
    p=R/f
    try: src=p.read_text(); out=ast.unparse(ast.parse(src))+"\n"
    except Exception: continue
    if out!=src: p.write_text(out); n+=1
git("commit","-qam","fmt-only")
v=view()
flagged=[s for s in v.symbols if s.change=="modified"]
print(f"formatting-only: {n} files rewritten; modified symbols reported: {len(flagged)}; calls changed: {len(v.calls)}; claims: {len(v.claims())}")
for s in flagged[:5]: print("   !!", s.qualname, sorted(s.flags))
# 2. mutation: change one integer constant in 40 random functions
random.seed(7); cands=[]
for f in files:
    try: tree=ast.parse((R/f).read_text())
    except Exception: continue
    for fn in ast.walk(tree):
        if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)) and not any(isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for x in ast.walk(fn) if x is not fn):
            if any(isinstance(c,ast.Constant) and type(c.value) is int and not isinstance(c.value,bool) for c in ast.walk(fn)):
                cands.append((f,fn.name,fn.lineno))
random.shuffle(cands); picks=[]; seen=set()
for f,name,line in cands:
    if f in seen: continue
    seen.add(f); picks.append((f,name,line))
    if len(picks)==40: break
mut=[]
for f,name,line in picks:
    p=R/f; tree=ast.parse(p.read_text())
    fn=[x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name and x.lineno==line][0]
    for c in ast.walk(fn):
        if isinstance(c,ast.Constant) and type(c.value) is int and not isinstance(c.value,bool): c.value+=1000; break
    p.write_text(ast.unparse(tree)+"\n"); mut.append((str(f),name))
git("commit","-qam","mutations")
v=view()
body={(s.file,s.qualname.rsplit(".",1)[-1]) for s in v.symbols if Flag.BODY_CHANGED in s.flags}
hit=sum(1 for m in mut if (m[0],m[1]) in body)
extra=[s for s in v.symbols if Flag.BODY_CHANGED in s.flags and (s.file,s.qualname.rsplit(".",1)[-1]) not in set(mut)]
print(f"mutation: {len(mut)} functions edited; BODY_CHANGED reported on {hit}; BODY_CHANGED on others (enclosing functions): {len(extra)}")
print("missed:", [m for m in mut if m not in body])
