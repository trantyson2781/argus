"""Independent cross-check of `argus explore diff` on real commits. Uses only ast/tokenize/git, never Argus's comparison code."""
import ast, io, subprocess, sys, tokenize
from pathlib import Path

from argus.explore.changes import Flag, build_diff_view
from argus.explore.sides import materialize_sides
from argus.extract import scan
from argus.extract.resolve_calls import resolve_v2_enabled

FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True).stdout


def show(repo, rev, f):
    try:
        return git(repo, "show", f"{rev}:{f}").decode()
    except subprocess.CalledProcessError:
        return None


def defs(src):
    """{in-file qualname: (kind, node)} -- independent walker."""
    out = {}
    def walk(body, prefix, in_class):
        for n in body:
            if isinstance(n, FUNCS + (ast.ClassDef,)):
                q = prefix + n.name
                kind = "class" if isinstance(n, ast.ClassDef) else ("method" if in_class else "function")
                out.setdefault(q, []).append((kind, n))
                walk(n.body, q + ".", isinstance(n, ast.ClassDef))
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(n, attr, [])
                    walk([s for h in sub for s in (h.body if isinstance(h, ast.ExceptHandler) else [h])], prefix, in_class)
    walk(ast.parse(src).body, "", False)
    return out


def body_tokens(src, fn):
    lo = fn.body[0].lineno
    hi = fn.end_lineno
    toks = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if not (lo <= t.start[0] <= hi):
            continue
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        s = t.string
        if t.type == tokenize.STRING:
            try:
                s = repr(ast.literal_eval(s))
            except Exception:
                s = s
        toks.append(s)
    return toks


def iface(fn):
    return ast.dump(fn.args) + "|" + "|".join(ast.dump(d) for d in fn.decorator_list) + "|" + ast.dump(fn.returns) if fn.returns else ast.dump(fn.args) + "|" + "|".join(ast.dump(d) for d in fn.decorator_list) + "|" + type(fn).__name__


def audit(repo, commit):
    repo = Path(repo)
    parent = commit + "~1"
    changed = [f for f in git(repo, "diff", "--name-only", parent, commit).decode().split() if f.endswith(".py")]
    rv2 = resolve_v2_enabled()
    with materialize_sides(repo, parent, commit) as (b, h):
        view = build_diff_view(b, h, scan(b.root, resolve_v2=rv2), scan(h.root, resolve_v2=rv2), rv2, True)
        mine = {}  # (file, suffix) -> flags / change
        for s in view.symbols:
            mine[(s.file, s.qualname)] = s
        issues, checked = [], 0
        for f in changed:
            old, new = show(repo, parent, f), show(repo, commit, f)
            od = defs(old) if old else {}
            nd = defs(new) if new else {}
            def find(q, file):
                for (sf, sq), s in mine.items():
                    if sf == file and sq.endswith("." + q):
                        return s
                    if sf == file and sq == q:
                        return s
            for q in nd.keys() - od.keys():
                kind = nd[q][0][0]; checked += 1
                s = find(q, f)
                if not s or s.change != "added":
                    issues.append(("MISSED ADD", f, q))
            for q in od.keys() - nd.keys():
                checked += 1
                s = find(q, f)
                if not s or s.change != "removed":
                    issues.append(("MISSED REMOVE", f, q))
            for q in od.keys() & nd.keys():
                if len(od[q]) > 1 or len(nd[q]) > 1:
                    continue
                ok, nk = od[q][0], nd[q][0]
                if not isinstance(nk[1], FUNCS) or not isinstance(ok[1], FUNCS):
                    continue
                checked += 1
                body_diff = body_tokens(old, ok[1]) != body_tokens(new, nk[1])
                sig_diff = iface(ok[1]) != iface(nk[1])
                s = find(q, f)
                got_body = bool(s and Flag.BODY_CHANGED in s.flags)
                got_sig = bool(s and Flag.SIGNATURE_CHANGED in s.flags)
                if body_diff != got_body:
                    issues.append(("BODY MISMATCH argus=%s indep=%s" % (got_body, body_diff), f, q))
                if sig_diff != got_sig:
                    issues.append(("SIG MISMATCH argus=%s indep=%s" % (got_sig, sig_diff), f, q))
        # extra argus symbols for changed files that independent walker doesn't know
        for (sf, sq), s in mine.items():
            if s.change == "modified" and not s.flags:
                issues.append(("MODIFIED WITHOUT FLAGS", sf, sq))
        # evidence text check: caller line must contain the callee's last name segment
        ev_total = ev_bad = 0
        bad = []
        for g in (*view.interface_callers, *view.affected_callers):
            for e in g.edges:
                ev = e.claim.evidence[0]
                root = b.root if e.was else h.root
                try:
                    line = (root / ev.file).read_text().splitlines()[ev.line - 1]
                except Exception as ex:
                    line = ""
                ev_total += 1
                name = e.callee.rsplit(".", 1)[-1]
                if name not in line:
                    ev_bad += 1; bad.append((g.caller, e.callee, ev.location, line.strip()[:80]))
        counts = dict(files=view.files_changed, symbols=len(view.symbols), calls=len(view.calls),
                      iface_groups=len(view.interface_callers), aff_groups=len(view.affected_callers),
                      edges=ev_total, edge_text_bad=ev_bad, claims=len(view.claims()))
        calls_sample = [(c.change, c.source, c.target, c.claim.evidence[0].location,
                         ((b.root if c.change == "removed" else h.root) / c.claim.evidence[0].file).read_text().splitlines()[c.claim.evidence[0].line - 1].strip()[:90]) for c in view.calls]
        return counts, checked, issues, bad, calls_sample, view


if __name__ == "__main__":
    repo, commits = sys.argv[1], sys.argv[2:]
    for c in commits:
        counts, checked, issues, bad, calls, view = audit(repo, c)
        print(f"== {repo.split('/')[-1]} {c}: {counts}; independent checks={checked}; disagreements={len(issues)}")
        for i in issues[:15]:
            print("   ", i)
        for x in bad[:8]:
            print("    EVIDENCE-TEXT?", x)
        for k in calls[:6]:
            print("    CALL", k)
