"""Choose the regional stations by RULE (geography + archive coverage only; no model result is ever looked at).

    python experiments/regional_cities/select_stations.py     # reads archive_coverage.json, writes stations.json

archive_coverage.json = every PM2.5 station of the public OpenAQ archive (legacy locationid layout) inside lat 27.7-32, lon 74-78
with the number of daily files per year (made by discover_stations.py).

Rules (declared in config.json):
  * Delhi stations are NEVER regional: the Delhi city mean (the target) must stay Delhi only.
  * only government monitoring networks: names ending in HSPCB, UPPCB, PPCB, RSPCB or CPCC
  * coverage: >= 365 daily files in 2018-2022 AND >= 150 in 2025-26 (so the station exists in both the training years and the test years)
  * group by distance / bearing from central Delhi (28.6139 N, 77.2090 E)
"""
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DELHI = (28.6139, 77.2090)
BOARDS = ("HSPCB", "UPPCB", "PPCB", "RSPCB", "CPCC")
NOT_REGIONAL_NAMES = ("delhi", "igi airport", "civil lines-", "us diplomatic", "income tax office")   # Delhi stations without "Delhi" in the name


def haversine_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1]); dp = p2 - p1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def bearing_deg(a, b):
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def board(name):
    m = re.search(r"\b(HSPCB|UPPCB|PPCB|RSPCB|CPCC)\b", name or "")
    return m.group(1) if m else None


def group_of(dist, bearing, b):
    if dist <= 55:
        return "ncr_near"                                   # inside the Delhi airshed: Noida, Ghaziabad, Gurugram, Faridabad, Bahadurgarh, Sonipat...
    if b == "PPCB":
        return "punjab"                                     # the stubble-burning region (Ludhiana, Amritsar, Patiala, Jalandhar, Bathinda...)
    if b == "UPPCB":
        return "up_east"                                    # Meerut, Hapur, Bulandshahr, Muzaffarnagar, Baghpat
    if b in ("HSPCB", "CPCC") and 250 <= bearing <= 360 and dist <= 300:
        return "haryana_nw"                                 # Panipat, Karnal, Kurukshetra, Rohtak, Jind, Hisar, Ambala, Chandigarh...
    return None                                             # south (Rajasthan/Palwal...) etc.: not used


def main():
    cov = json.loads((HERE / "archive_coverage.json").read_text())
    out, dropped = [], []
    for s in cov:
        name = s["name"] or ""
        low = name.lower()
        why = None
        if any(k in low for k in NOT_REGIONAL_NAMES):
            why = "Delhi"
        elif board(name) is None:
            why = "not a government network station (HSPCB/UPPCB/PPCB/RSPCB/CPCC)"
        else:
            by = s["by_year"]
            early = sum(by.get(str(y), 0) for y in range(2018, 2023)); late = by.get("2025", 0) + by.get("2026", 0)
            if early < 365 or late < 150:
                why = f"coverage {early} days in 2018-22, {late} in 2025-26"
        if why:
            dropped.append({"id": s["id"], "name": name, "why": why}); continue
        d = haversine_km(DELHI, (s["lat"], s["lon"])); br = bearing_deg(DELHI, (s["lat"], s["lon"]))
        g = group_of(d, br, board(name))
        if g is None:
            dropped.append({"id": s["id"], "name": name, "why": f"outside every group (dist {d:.0f} km, bearing {br:.0f})"}); continue
        out.append({"id": s["id"], "name": name, "lat": s["lat"], "lon": s["lon"], "board": board(name), "dist_km": round(d, 1),
                    "bearing": round(br), "group": g, "files_2018_22": sum(s["by_year"].get(str(y), 0) for y in range(2018, 2023)),
                    "files_2025_26": s["by_year"].get("2025", 0) + s["by_year"].get("2026", 0)})
    out.sort(key=lambda x: (x["group"], x["dist_km"]))
    (HERE / "stations.json").write_text(json.dumps({"rules": __doc__.split("Rules")[1].strip(), "stations": out, "dropped": dropped}, indent=1) + "\n")
    from collections import Counter
    print(Counter(x["group"] for x in out), "kept;", len(dropped), "dropped")
    for g in ("ncr_near", "haryana_nw", "punjab", "up_east"):
        print(f"\n== {g}")
        for x in out:
            if x["group"] == g:
                print(f"  {x['id']:>6} {x['dist_km']:>6.0f} km {x['bearing']:>3}°  {x['files_2018_22']:>4}/{x['files_2025_26']:>3}  {x['name'][:52]}")
    print("\n== dropped (not Delhi):")
    for d in dropped:
        if d["why"] != "Delhi":
            print("  ", d["id"], d["name"][:50], "-", d["why"])


if __name__ == "__main__":
    main()
