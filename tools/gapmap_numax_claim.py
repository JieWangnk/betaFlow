"""Claim test (gap-map step 6) for the nuMax reporting claim.

CLAIM UNDER TEST: "the regularisation parameter is an unreported degree of
freedom in Casson haemodynamics."

Counts are from TITLE+ABSTRACT metadata only. That is a hard recall limit and
it cuts one way: a regularisation cap is a methods detail that almost never
appears in an abstract, so a low co-mention count is NOT proof of
non-reporting. Absolute counts are lower bounds; relative comparisons against
a control pair are what carry information.
"""
import json, re, time, urllib.parse, urllib.request, pathlib

OUT = pathlib.Path(__file__).parent
UA = "betaflow-gapmap/0.1 (mailto:jieandwang@gmail.com)"


def crossref(q, rows=200, filt="from-pub-date:2000-01-01"):
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(q) + f"&rows={rows}&filter={filt}&select=DOI,title,abstract,container-title,issued")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return [{"doi": i.get("DOI"), "title": " ".join(i.get("title") or []),
             "abstract": re.sub(r"<[^>]+>", " ", i.get("abstract") or ""),
             "venue": " ".join(i.get("container-title") or []),
             "year": (i.get("issued", {}).get("date-parts") or [[None]])[0][0],
             "src": "crossref"} for i in d["message"]["items"]]


def pubmed(q, retmax=200):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    u = base + "esearch.fcgi?db=pubmed&retmode=json&retmax=%d&term=%s" % (retmax, urllib.parse.quote(q))
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=60) as r:
        ids = json.load(r)["esearchresult"].get("idlist", [])
    out = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i+100]
        if not chunk:
            break
        u2 = base + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ",".join(chunk)
        with urllib.request.urlopen(urllib.request.Request(u2, headers={"User-Agent": UA}), timeout=90) as r:
            xml = r.read().decode("utf-8", "ignore")
        for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
            t = " ".join(re.findall(r"<ArticleTitle[^>]*>(.*?)</ArticleTitle>", art, re.S))
            a = " ".join(re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", art, re.S))
            v = " ".join(re.findall(r"<Title>(.*?)</Title>", art, re.S))
            y = (re.findall(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S) or [None])[0]
            out.append({"doi": None, "title": re.sub(r"<[^>]+>", " ", t),
                        "abstract": re.sub(r"<[^>]+>", " ", a), "venue": v,
                        "year": int(y) if y else None, "src": "pubmed"})
        time.sleep(0.4)
    return out


QUERIES = {
    # pillars
    "casson_blood": ["Casson blood flow simulation", "Casson model artery hemodynamics",
                     "Casson viscosity blood numerical"],
    "regularisation_viscoplastic": ["regularisation viscoplastic yield stress flow",
                                    "Papanastasiou regularisation yield stress",
                                    "bi-viscosity model yield stress simulation"],
    # bridge — the intersection the claim is about
    "bridge_casson_regularisation": ["Casson blood regularisation parameter",
                                     "yield stress blood flow regularisation numerical",
                                     "Casson hemodynamics Papanastasiou"],
    # CONTROL: a pair that SHOULD overlap, so a low bridge count means something
    "control_casson_wss": ["Casson blood wall shear stress simulation",
                           "non-Newtonian blood wall shear stress artery"],
}

state = {}
for tag, qs in QUERIES.items():
    recs = []
    for q in qs:
        for fn, kw in ((crossref, {}), (pubmed, {})):
            try:
                recs += fn(q, **kw)
            except Exception as e:
                print("FAIL", tag, q, fn.__name__, repr(e)[:80], flush=True)
        time.sleep(0.5)
    state[tag] = recs
    print("%-30s %d records" % (tag, len(recs)), flush=True)

# dedup on normalised title
seen, corpus = set(), []
for tag, recs in state.items():
    for r in recs:
        k = re.sub(r"\W+", "", (r["title"] or "").lower())[:90]
        if not k or k in seen:
            continue
        seen.add(k)
        r["query_tag"] = tag
        corpus.append(r)
json.dump(corpus, open(OUT / "corpus.json", "w"))
print("deduplicated corpus: %d" % len(corpus), flush=True)
