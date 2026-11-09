#!/usr/bin/env python3
"""Download the open data used by REACT-PF into data_cache/ and verify SHA-256 checksums.

The cached files shipped with the repository were downloaded on 5 November 2026 from:
  GridKit extract (PyPSA-Eur)   https://raw.githubusercontent.com/PyPSA/pypsa-eur/master/data/entsoegridkit/
  powerplantmatching            https://raw.githubusercontent.com/PyPSA/powerplantmatching/master/powerplants.csv
  SciGRID-DE, model-energy      https://raw.githubusercontent.com/PyPSA/PyPSA/master/examples/networks/
Upstream files may change; the checksums in SHA256SUMS identify the versions used in the paper.
Usage: python fetch_data.py [--force]
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data_cache")
GK = "https://raw.githubusercontent.com/PyPSA/pypsa-eur/master/data/entsoegridkit/"
EX = "https://raw.githubusercontent.com/PyPSA/PyPSA/master/examples/networks/"
FILES = {
    "gridkit_buses.csv": GK + "buses.csv", "gridkit_lines.csv": GK + "lines.csv",
    "gridkit_links.csv": GK + "links.csv", "gridkit_transformers.csv": GK + "transformers.csv",
    "powerplants.csv": "https://raw.githubusercontent.com/PyPSA/powerplantmatching/master/powerplants.csv",
    "scigrid-de.nc": EX + "scigrid-de/scigrid-de.nc", "model-energy.nc": EX + "model-energy/model-energy.nc",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(force=False):
    os.makedirs(CACHE, exist_ok=True)
    ref = {}
    sums = os.path.join(CACHE, "SHA256SUMS")
    if os.path.exists(sums):
        for line in open(sums):
            h, name = line.split()
            ref[name] = h
    for name, url in FILES.items():
        path = os.path.join(CACHE, name)
        if force or not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(url, path)
        h = sha(path)
        status = "ok" if ref.get(name) == h else ("NEW VERSION (differs from paper)" if name in ref else "no reference")
        print(f"{name}: {status}")


if __name__ == "__main__":
    main("--force" in sys.argv)
