#!/usr/bin/env python3
"""
Maco Explorer – radar SHMÚ pre appku.

Každých 5 minút (GitHub Actions alebo VPS cez cron):
  1. nájde najnovší HDF5 súbor kompozitu SHMÚ (CAPPI 2 km),
  2. prečíta odrazivosť (dBZ) cez h5py, vyfarbí ju (pod 10 dBZ priehľadné),
  3. prepočíta obrázok do EPSG:3857 (Web Mercator, ako mapy v telefóne) a uloží ako WebP,
  4. nahrá ho na Cloudflare R2, zmaže obrázky staršie ako 3 hodiny,
  5. zapíše manifest.json s poslednými 24 snímkami (čas, URL) a ohraničením (bbox).

Radarové dáta: SHMÚ OpenData (https://opendata.shmu.sk).

Premenné prostredia (v GitHub Actions ako Secrets):
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET,
  R2_PUBLIC_BASE   napr. https://radar.macoexplorer.sk  (verejná adresa bucketu, bez lomky na konci)

Lokálny test bez R2:   python shmu_radar.py --local out/      (alebo --file stiahnuty.hdf)
"""
import argparse, datetime as dt, io, json, os, re, sys, urllib.request

import h5py
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer

BASE = "https://opendata.shmu.sk/meteorology/weather/radar/composite/skcomp/cappi2km/"
KEEP_FRAMES = 24          # v manifeste (2 hodiny)
DELETE_AFTER_H = 3        # obrázky staršie ako toto sa zmažú
UA = "MacoExplorerRadar/1.0 (+https://macoexplorer.sk)"

# Farby podľa dBZ (od, R, G, B, A). Pod 10 dBZ nič (priehľadné).
COLORS = [
    (10, 126, 200, 242, 150),   # slabý dážď
    (20, 46, 123, 214, 190),
    (28, 46, 184, 75, 210),
    (35, 244, 208, 63, 225),    # mierny
    (42, 230, 126, 34, 235),
    (48, 192, 57, 43, 245),     # silný
    (55, 142, 68, 173, 250),    # búrka, krupobitie
]


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def latest_file_url(now=None):
    """Najnovší súbor v priečinku dnešného (alebo včerajšieho) dňa."""
    now = now or dt.datetime.now(dt.timezone.utc)
    for day in (now, now - dt.timedelta(days=1)):
        folder = BASE + day.strftime("%Y%m%d") + "/"
        try:
            html = http_get(folder).decode("utf-8", "replace")
        except Exception as e:  # priečinok ešte nemusí existovať
            print(f"priečinok {folder}: {e}", file=sys.stderr)
            continue
        names = sorted(set(re.findall(r'href="([^"?/]+\.(?:hdf5?|h5))"', html, re.I)))
        if names:
            return folder + names[-1]
    raise SystemExit("Nenašiel som žiadny HDF5 súbor radaru SHMÚ.")


def attr(group, name, default=None):
    v = group.attrs.get(name, default) if group is not None else default
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.item()
        if isinstance(v, bytes):
            v = v.decode()
    return v


def read_odim(raw: bytes):
    """Vráti (dBZ ako float pole s NaN, CRS zdroja, x0, y0, xscale, yscale, čas UTC)."""
    with h5py.File(io.BytesIO(raw), "r") as f:
        ds = f["dataset1"]
        data_grp = ds["data1"] if "data1" in ds else ds
        arr = data_grp["data"][()].astype(np.float64)
        w_data = data_grp.get("what")
        w_ds = ds.get("what")
        def what(name, default):
            v = attr(w_data, name, None)
            if v is None:
                v = attr(w_ds, name, None)
            return default if v is None else v
        gain, offset = float(what("gain", 1.0)), float(what("offset", 0.0))
        nodata, undetect = what("nodata", None), what("undetect", None)
        mask = np.zeros(arr.shape, bool)
        if nodata is not None:
            mask |= arr == float(nodata)
        if undetect is not None:
            mask |= arr == float(undetect)
        dbz = arr * gain + offset
        dbz[mask] = np.nan

        where = f["where"]
        crs = CRS.from_proj4(attr(where, "projdef"))
        ysize, xsize = arr.shape
        to_src = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        ul = to_src.transform(float(attr(where, "UL_lon")), float(attr(where, "UL_lat")))
        lr_lon, lr_lat = attr(where, "LR_lon"), attr(where, "LR_lat")
        if lr_lon is not None:
            lr = to_src.transform(float(lr_lon), float(lr_lat))
            xscale, yscale = (lr[0] - ul[0]) / xsize, (ul[1] - lr[1]) / ysize
        else:
            xscale, yscale = float(attr(where, "xscale")), float(attr(where, "yscale"))

        what_root = f["what"]
        d, t = str(attr(what_root, "date")), str(attr(what_root, "time")).ljust(6, "0")
        when = dt.datetime.strptime(d + t[:6], "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    return dbz, crs, ul[0], ul[1], xscale, yscale, when


def colorize(dbz):
    rgba = np.zeros(dbz.shape + (4,), np.uint8)
    valid = ~np.isnan(dbz)
    for lo, r, g, b, a in COLORS:
        m = valid & (dbz >= lo)
        rgba[m] = (r, g, b, a)
    return rgba


def to_web_mercator(dbz, crs, x0, y0, xs, ys):
    """Nearest-neighbour prevzorkovanie do EPSG:3857. Vráti (RGBA obrázok, bbox [W, S, E, N])."""
    h, w = dbz.shape
    src_to_merc = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    # okraj zdroja (husto), aby sa zachytilo zakrivenie
    t = np.linspace(0, 1, 200)
    xs_edge = np.concatenate([x0 + t * w * xs, np.full_like(t, x0 + w * xs), x0 + t * w * xs, np.full_like(t, x0)])
    ys_edge = np.concatenate([np.full_like(t, y0), y0 - t * h * ys, np.full_like(t, y0 - h * ys), y0 - t * h * ys])
    mx, my = src_to_merc.transform(xs_edge, ys_edge)
    minx, maxx, miny, maxy = mx.min(), mx.max(), my.min(), my.max()
    out_w = w
    out_h = max(1, int(round(out_w * (maxy - miny) / (maxx - minx))))
    gx = minx + (np.arange(out_w) + 0.5) * (maxx - minx) / out_w
    gy = maxy - (np.arange(out_h) + 0.5) * (maxy - miny) / out_h
    GX, GY = np.meshgrid(gx, gy)
    merc_to_src = Transformer.from_crs("EPSG:3857", crs, always_xy=True)
    sx, sy = merc_to_src.transform(GX, GY)
    col = np.floor((sx - x0) / xs).astype(np.int64)
    row = np.floor((y0 - sy) / ys).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out = np.full((out_h, out_w), np.nan)
    out[inside] = dbz[row[inside], col[inside]]
    merc_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    west, south = merc_to_ll.transform(minx, miny)
    east, north = merc_to_ll.transform(maxx, maxy)
    img = Image.fromarray(colorize(out), "RGBA")
    return img, [round(west, 5), round(south, 5), round(east, 5), round(north, 5)]


def encode(img, fmt):
    buf = io.BytesIO()
    if fmt == "webp":
        img.save(buf, "WEBP", lossless=True, method=4)
    else:
        img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def update_manifest(old, frame, bbox, now):
    frames = [f for f in (old or {}).get("frames", []) if f.get("time") != frame["time"]]
    frames.append(frame)
    cutoff = now - dt.timedelta(hours=DELETE_AFTER_H)
    frames = [f for f in frames if dt.datetime.strptime(f["time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) >= cutoff]
    frames.sort(key=lambda f: f["time"])
    return {
        "source": "SHMÚ OpenData",
        "attribution": "Radarové dáta: SHMÚ OpenData",
        "product": "skcomp cappi2km dBZ",
        "updated": iso(now),
        "bbox": bbox,
        "frames": frames[-KEEP_FRAMES:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="lokálny HDF5 súbor namiesto sťahovania")
    ap.add_argument("--local", help="zapísať do priečinka namiesto R2")
    ap.add_argument("--format", default="webp", choices=["webp", "png"])
    a = ap.parse_args()

    if a.file:
        raw = open(a.file, "rb").read()
        print("súbor:", a.file)
    else:
        url = latest_file_url()
        print("sťahujem:", url)
        raw = http_get(url, timeout=60)

    dbz, crs, x0, y0, xs, ys, when = read_odim(raw)
    img, bbox = to_web_mercator(dbz, crs, x0, y0, xs, ys)
    body = encode(img, a.format)
    key = f"frames/{when.strftime('%Y%m%d%H%M')}.{a.format}"
    now = dt.datetime.now(dt.timezone.utc)
    ctype = "image/webp" if a.format == "webp" else "image/png"
    print(f"čas merania {iso(when)}, obrázok {img.size[0]}×{img.size[1]}, {len(body)//1024} kB, bbox {bbox}")

    if a.local:
        os.makedirs(os.path.join(a.local, "frames"), exist_ok=True)
        open(os.path.join(a.local, key), "wb").write(body)
        mpath = os.path.join(a.local, "manifest.json")
        old = json.load(open(mpath)) if os.path.exists(mpath) else None
        base = os.environ.get("R2_PUBLIC_BASE", "https://radar.example.sk")
        m = update_manifest(old, {"time": iso(when), "url": f"{base}/{key}"}, bbox, max(now, when))
        json.dump(m, open(mpath, "w"), ensure_ascii=False, indent=1)
        print("zapísané do", a.local)
        return

    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket, base = os.environ["R2_BUCKET"], os.environ["R2_PUBLIC_BASE"].rstrip("/")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=ctype,
                  CacheControl="public, max-age=86400, immutable")
    try:
        old = json.loads(s3.get_object(Bucket=bucket, Key="manifest.json")["Body"].read())
    except Exception:
        old = None
    m = update_manifest(old, {"time": iso(when), "url": f"{base}/{key}"}, bbox, max(now, when))
    s3.put_object(Bucket=bucket, Key="manifest.json", Body=json.dumps(m, ensure_ascii=False).encode(),
                  ContentType="application/json; charset=utf-8", CacheControl="public, max-age=60")

    # upratanie: obrázky staršie ako DELETE_AFTER_H hodín
    cutoff = now - dt.timedelta(hours=DELETE_AFTER_H)
    keep = {f["url"].rsplit("/", 2)[-2] + "/" + f["url"].rsplit("/", 1)[-1] for f in m["frames"]}
    old_keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="frames/"):
        for o in page.get("Contents", []):
            if o["Key"] not in keep and o["LastModified"] < cutoff:
                old_keys.append({"Key": o["Key"]})
    for i in range(0, len(old_keys), 1000):
        s3.delete_objects(Bucket=bucket, Delete={"Objects": old_keys[i:i + 1000]})
    print(f"nahrané {key}, v manifeste {len(m['frames'])} snímok, zmazaných {len(old_keys)}")


if __name__ == "__main__":
    main()
