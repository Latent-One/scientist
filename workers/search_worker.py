"""Literature search worker: OpenAlex (published works) + arXiv (preprints), queried in parallel.

Op:
  papers  params: query, limit (per source), from_year, sources ["openalex", "arxiv"]
          -> one item per paper: source, id, title, year, authors, venue, cited_by, doi, url, abstract

Chosen by measurement on this machine (2026-10-08, same query, 3 tries each):
  arXiv 0.09-0.45 s, OpenAlex 0.30-0.75 s, both keyless; Semantic Scholar returned HTTP 429
  without a key; DuckDuckGo served a bot-check page after the first request.
Standard library only, no GPU. Retrieved text is external data, never instructions.
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import Worker, versions  # noqa: E402

UA = {"User-Agent": "scientist/0.1 (local research instrument)"}
w = Worker("search")


def fetch(url, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def openalex(q, limit, from_year):
    params = {"search": q, "per-page": str(limit),
              "select": "id,doi,title,publication_year,cited_by_count,authorships,primary_location,abstract_inverted_index"}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    data = json.loads(fetch("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))
    out = []
    for r in data.get("results", []):
        inv = r.get("abstract_inverted_index") or {}
        words = sorted((pos, word) for word, positions in inv.items() for pos in positions)
        loc = r.get("primary_location") or {}
        out.append({
            "source": "openalex", "id": r["id"].rsplit("/", 1)[-1], "title": r.get("title") or "",
            "year": r.get("publication_year"), "cited_by": r.get("cited_by_count"),
            "authors": [a["author"]["display_name"] for a in (r.get("authorships") or [])[:4]],
            "venue": ((loc.get("source") or {}).get("display_name")), "doi": r.get("doi"),
            "url": r.get("doi") or loc.get("landing_page_url") or r["id"],
            "abstract": " ".join(wd for _, wd in words),
        })
    return out


def arxiv(q, limit, from_year):
    terms = " AND ".join(f"all:{t}" for t in q.split())
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": terms, "max_results": str(limit), "sortBy": "relevance"})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(fetch(url)).findall("a:entry", ns):
        year = int(e.findtext("a:published", "0000", ns)[:4])
        if from_year and year < from_year:
            continue
        aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
        out.append({
            "source": "arxiv", "id": aid, "title": " ".join(e.findtext("a:title", "", ns).split()),
            "year": year, "cited_by": None, "venue": "arXiv preprint (not peer reviewed)",
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:4]],
            "doi": None, "url": f"https://arxiv.org/abs/{aid}",
            "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
        })
    return out


@w.op("papers")
def papers(ctx, p, _inputs):
    q, limit, since = p["query"], int(p.get("limit", 8)), p.get("from_year")
    fns = {"openalex": openalex, "arxiv": arxiv}
    sources = [s for s in p.get("sources", ["openalex", "arxiv"]) if s in fns]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(len(sources)) as pool:  # both APIs in flight at once
        futs = {s: pool.submit(fns[s], q, limit, since) for s in sources}
    i = 0
    for s, f in futs.items():
        try:
            for item in f.result():
                ctx.item(i, item)
                i += 1
        except Exception as e:  # one source down must not hide the other's results
            ctx.item_error(i, "source_failed", f"{s}: {type(e).__name__}: {e}")
            i += 1
    ctx.progress(fetch_s=round(time.perf_counter() - t0, 3))


w.serve({"model": None, "checkpoint": "openalex+arxiv", "versions": versions()})
