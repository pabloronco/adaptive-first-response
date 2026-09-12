from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path


WIDTH = 1400
HEIGHT = 1000
PADDING = 80


def _pick(row: dict, *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in ("", None):
            return row[name]
    return default


def load_sites(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            site_id = _pick(row, "site_id", "SiteID", "site")
            lat = float(_pick(row, "latitude", "lat"))
            lon = float(_pick(row, "longitude", "lon", "lng"))
            rows.append(
                {
                    "site_id": site_id,
                    "latitude": lat,
                    "longitude": lon,
                }
            )
        return rows


def load_edges(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            src = _pick(row, "src", "source", "u")
            dst = _pick(row, "dst", "target", "v")
            if src and dst:
                rows.append({"src": src, "dst": dst})
        return rows


def bbox_for_sites(sites: list[dict], pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    lats = [s["latitude"] for s in sites]
    lons = [s["longitude"] for s in sites]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_pad = max((max_lat - min_lat) * pad_ratio, 0.03)
    lon_pad = max((max_lon - min_lon) * pad_ratio, 0.03)
    return (
        min_lon - lon_pad,
        min_lat - lat_pad,
        max_lon + lon_pad,
        max_lat + lat_pad,
    )


def project(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    x = PADDING + (lon - min_lon) / (max_lon - min_lon) * (WIDTH - 2 * PADDING)
    y = HEIGHT - (
        PADDING + (lat - min_lat) / (max_lat - min_lat) * (HEIGHT - 2 * PADDING)
    )
    return x, y


def try_load_local_coastlines(bbox: tuple[float, float, float, float]) -> tuple[list[list[tuple[float, float]]], str]:
    """
    Cerca localmente eventuali geojson/json con shoreline/shorezone/coastline.
    Se li trova, estrae LineString/MultiLineString e li filtra sul bbox dei siti.
    """
    patterns = [
        "data/**/*shorezone*.geojson",
        "data/**/*shorezone*.json",
        "data/**/*shoreline*.geojson",
        "data/**/*shoreline*.json",
        "data/**/*coast*.geojson",
        "data/**/*coast*.json",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    min_lon, min_lat, max_lon, max_lat = bbox
    segments: list[list[tuple[float, float]]] = []

    for filename in sorted(set(candidates)):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        features = []
        if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
            features = obj.get("features", [])
        elif isinstance(obj, dict) and obj.get("type") == "Feature":
            features = [obj]
        else:
            continue

        for feat in features:
            geom = feat.get("geometry") or {}
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "LineString":
                line = [(float(x), float(y)) for x, y in coords]
                if line_intersects_bbox(line, bbox):
                    segments.append(line)
            elif gtype == "MultiLineString":
                for part in coords:
                    line = [(float(x), float(y)) for x, y in part]
                    if line_intersects_bbox(line, bbox):
                        segments.append(line)

    if segments:
        return segments, "local_geojson"
    return [], "none"


def line_intersects_bbox(line: list[tuple[float, float]], bbox: tuple[float, float, float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    for lon, lat in line:
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return True
    return False


def build_components(sites: list[dict], edges: list[dict]) -> list[list[str]]:
    site_ids = {s["site_id"] for s in sites}
    adj = {sid: set() for sid in site_ids}
    for e in edges:
        if e["src"] in adj and e["dst"] in adj:
            adj[e["src"]].add(e["dst"])
            adj[e["dst"]].add(e["src"])

    seen = set()
    comps = []
    for sid in sorted(site_ids):
        if sid in seen:
            continue
        stack = [sid]
        comp = []
        seen.add(sid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def fallback_coast_trace(sites: list[dict], edges: list[dict]) -> list[list[tuple[float, float]]]:
    """
    Fallback: se non troviamo geometrie costiere locali,
    tracciamo una linea schematica per componente, seguendo i siti reali.
    NON è shoreline vera: è solo contesto visivo.
    """
    by_id = {s["site_id"]: s for s in sites}
    components = build_components(sites, edges)
    traces = []

    for comp in components:
        pts = [by_id[sid] for sid in comp]
        if len(pts) == 1:
            traces.append([(pts[0]["longitude"], pts[0]["latitude"])])
            continue

        # start from westernmost point
        remaining = pts[:]
        current = min(remaining, key=lambda p: (p["longitude"], p["latitude"]))
        path = [current]
        remaining.remove(current)

        while remaining:
            nxt = min(
                remaining,
                key=lambda p: haversine_km(
                    current["latitude"], current["longitude"],
                    p["latitude"], p["longitude"]
                ),
            )
            path.append(nxt)
            remaining.remove(nxt)
            current = nxt

        traces.append([(p["longitude"], p["latitude"]) for p in path])

    return traces


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def degrees_from_edges(sites: list[dict], edges: list[dict]) -> dict[str, int]:
    deg = {s["site_id"]: 0 for s in sites}
    for e in edges:
        if e["src"] in deg:
            deg[e["src"]] += 1
        if e["dst"] in deg:
            deg[e["dst"]] += 1
    return deg


def render_svg(
    sites: list[dict],
    edges: list[dict],
    coastlines: list[list[tuple[float, float]]],
    coastline_mode: str,
    out_path: Path,
) -> None:
    bbox = bbox_for_sites(sites)
    by_id = {s["site_id"]: s for s in sites}
    deg = degrees_from_edges(sites, edges)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    parts.append("<style>")
    parts.append("""
        .title { font: 700 28px Arial, sans-serif; fill: #0f172a; }
        .subtitle { font: 400 15px Arial, sans-serif; fill: #334155; }
        .legend { font: 400 14px Arial, sans-serif; fill: #334155; }
        .small { font: 400 12px Arial, sans-serif; fill: #475569; }
        .coast { fill: none; stroke: #94a3b8; stroke-width: 2; opacity: 0.9; }
        .coast-fallback { fill: none; stroke: #cbd5e1; stroke-width: 2.2; stroke-dasharray: 6 5; opacity: 0.95; }
        .edge { stroke: #64748b; stroke-width: 2; opacity: 0.40; }
        .node { fill: #0f766e; stroke: white; stroke-width: 1.5; }
        .node-isolated { fill: #d97706; stroke: white; stroke-width: 1.5; }
        .label { font: 600 12px Arial, sans-serif; fill: #0f172a; }
        .frame { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1.2; rx: 18; }
        .waterbg { fill: #f0f9ff; }
    """)
    parts.append("</style>")

    parts.append(f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" class="waterbg"/>')
    parts.append(f'<rect x="18" y="18" width="{WIDTH-36}" height="{HEIGHT-36}" class="frame"/>')

    parts.append(f'<text x="40" y="55" class="title">R2 Real Graph v0</text>')
    parts.append(
        '<text x="40" y="82" class="subtitle">Real monitoring sites + primary adjacency + coastal context</text>'
    )

    if coastline_mode == "local_geojson":
        parts.append(
            '<text x="40" y="104" class="small">Coast context source: local shoreline/shorezone geometry found in data/</text>'
        )
    else:
        parts.append(
            '<text x="40" y="104" class="small">Coast context source: schematic fallback trace from real site coordinates (visual aid only)</text>'
        )

    # coastlines
    for line in coastlines:
        if len(line) < 2:
            continue
        pts = [project(lon, lat, bbox) for lon, lat in line]
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        cls = "coast" if coastline_mode == "local_geojson" else "coast-fallback"
        parts.append(f'<polyline points="{points_attr}" class="{cls}"/>')

    # edges
    for e in edges:
        a = by_id.get(e["src"])
        b = by_id.get(e["dst"])
        if not a or not b:
            continue
        x1, y1 = project(a["longitude"], a["latitude"], bbox)
        x2, y2 = project(b["longitude"], b["latitude"], bbox)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge"/>')

    # nodes + labels
    for s in sites:
        x, y = project(s["longitude"], s["latitude"], bbox)
        cls = "node-isolated" if deg[s["site_id"]] == 0 else "node"
        radius = 6.8 if deg[s["site_id"]] == 0 else 5.8
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" class="{cls}"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" class="label">{s["site_id"]}</text>')

    # simple legend
    lx = WIDTH - 390
    ly = 55
    parts.append(f'<rect x="{lx}" y="{ly}" width="330" height="118" class="frame"/>')
    parts.append(f'<text x="{lx+18}" y="{ly+28}" class="legend">Legend</text>')
    parts.append(f'<line x1="{lx+18}" y1="{ly+50}" x2="{lx+58}" y2="{ly+50}" class="edge"/>')
    parts.append(f'<text x="{lx+70}" y="{ly+55}" class="legend">Primary graph edge</text>')
    parts.append(f'<circle cx="{lx+38}" cy="{ly+77}" r="6" class="node"/>')
    parts.append(f'<text x="{lx+70}" y="{ly+82}" class="legend">Connected site</text>')
    parts.append(f'<circle cx="{lx+38}" cy="{ly+102}" r="6.5" class="node-isolated"/>')
    parts.append(f'<text x="{lx+70}" y="{ly+107}" class="legend">Isolated site/component seed</text>')

    parts.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sites",
        default="data/processed/r2_real_sites.csv",
        help="CSV with site_id, latitude, longitude",
    )
    parser.add_argument(
        "--edges",
        default="data/processed/r2_real_graph_v0_edges.csv",
        help="CSV with src,dst",
    )
    parser.add_argument(
        "--out",
        default="reports/r2_real_graph_v0/real_graph_v0.svg",
        help="Output SVG path",
    )
    args = parser.parse_args()

    sites_path = Path(args.sites)
    edges_path = Path(args.edges)
    out_path = Path(args.out)

    if not sites_path.exists():
        raise SystemExit(f"Missing sites CSV: {sites_path}")
    if not edges_path.exists():
        raise SystemExit(f"Missing edges CSV: {edges_path}")

    sites = load_sites(sites_path)
    edges = load_edges(edges_path)
    bbox = bbox_for_sites(sites)

    coastlines, mode = try_load_local_coastlines(bbox)
    if not coastlines:
        coastlines = fallback_coast_trace(sites, edges)
        mode = "fallback_trace"

    render_svg(sites, edges, coastlines, mode, out_path)
    print(f"SVG written to: {out_path}")
    print(f"Coast mode: {mode}")
    print(f"Sites: {len(sites)} | Edges: {len(edges)} | Coast polylines: {len(coastlines)}")


if __name__ == "__main__":
    main()
