#!/usr/bin/env python3
"""Aggregate one public MaStR XML snapshot into German-state storage tables.

Usage: python tools/aggregate_mastr_storage.py export.zip --snapshot-date YYYY-MM-DD
       --output-dir data/mastr_storage/YYYY-MM-DD

Requires lxml. SQLite is used for the plant/unit join, avoiding a large RAM load.
The input is a single official Gesamtdatenexport, not a reconstructed time series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import time
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

from lxml import etree

STATE_ISO = {
    "Baden-Württemberg": "DE-BW", "Bayern": "DE-BY", "Berlin": "DE-BE",
    "Brandenburg": "DE-BB", "Bremen": "DE-HB", "Hamburg": "DE-HH",
    "Hessen": "DE-HE", "Mecklenburg-Vorpommern": "DE-MV",
    "Niedersachsen": "DE-NI", "Nordrhein-Westfalen": "DE-NW",
    "Rheinland-Pfalz": "DE-RP", "Saarland": "DE-SL", "Sachsen": "DE-SN",
    "Sachsen-Anhalt": "DE-ST", "Schleswig-Holstein": "DE-SH", "Thüringen": "DE-TH",
}


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def number(text):
    if text is None or not text.strip():
        return None
    value = float(text)
    return value if math.isfinite(value) and value >= 0 else None


def records(archive, names, tag, manifest):
    for name in names:
        info = archive.getinfo(name)
        count = 0
        with archive.open(name) as stream:
            reader = HashedReader(stream)
            for _, element in etree.iterparse(reader, events=("end",), tag=tag,
                                              resolve_entities=False, no_network=True):
                yield element
                count += 1
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
            # Reading to EOF also makes ZipExtFile complete its CRC check.
            while reader.read(1024 * 1024):
                pass
        manifest.append({"name": name, "records": count,
                         "uncompressed_bytes": info.file_size,
                         "compressed_bytes": info.compress_size,
                         "crc32": f"{info.CRC:08x}",
                         "uncompressed_sha256": reader.sha.hexdigest()})
        print(f"Read {name}: {count:,} records", flush=True)


def natural_key(name):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--snapshot-date", required=True, type=date.fromisoformat)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--battery-duration-flag-hours", type=float, default=8760.,
                        help="Flag battery energy/power ratios above this diagnostic threshold; raw totals remain intact")
    args = parser.parse_args()
    if not math.isfinite(args.battery_duration_flag_hours) or args.battery_duration_flag_hours <= 0:
        parser.error("--battery-duration-flag-hours must be finite and positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp = tempfile.TemporaryDirectory(prefix="mastr-storage-") if not args.work_dir else None
    work = args.work_dir or Path(temp.name)
    work.mkdir(parents=True, exist_ok=True)
    database = work / "storage.sqlite"
    if database.exists():
        raise FileExistsError(f"Use a fresh work directory; database exists: {database}")
    db = sqlite3.connect(database)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA cache_size=-524288; PRAGMA temp_store=MEMORY;
        CREATE TABLE units(id TEXT PRIMARY KEY, plant TEXT, state TEXT, technology TEXT,
                           net_kw REAL, gross_kw REAL) WITHOUT ROWID;
        CREATE TABLE plants(id TEXT PRIMARY KEY, energy_kwh REAL,
                            status TEXT) WITHOUT ROWID;
    """)
    counters = Counter()
    source_members = []
    foreign_plant_links = set()
    start = time.monotonic()
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        catalog_data = archive.read("Katalogwerte.xml")
        catalog = {}
        categories = {}
        for item in etree.fromstring(catalog_data):
            identifier = item.findtext("Id")
            catalog[identifier] = item.findtext("Wert") or ""
            categories[identifier] = item.findtext("KatalogKategorieId")
        source_members.append({"name": "Katalogwerte.xml",
                               "uncompressed_sha256": hashlib.sha256(catalog_data).hexdigest(),
                               "crc32": f"{archive.getinfo('Katalogwerte.xml').CRC:08x}"})
        lookup = lambda text, category: next(k for k, v in catalog.items()
                                             if v == text and categories[k] == category)
        germany = lookup("Deutschland", "6")
        operating = lookup("In Betrieb", "4")
        activated = lookup("Aktiviert", "19")
        battery = lookup("Batterie", "29")
        states = {k: v for k, v in catalog.items()
                  if categories[k] == "101" and v in STATE_ISO}
        if len(states) != 16:
            raise ValueError("The snapshot catalogue does not contain all 16 German states")
        technologies = {k: v for k, v in catalog.items() if categories[k] == "29"}
        unit_files = sorted([n for n in names if re.fullmatch(r"EinheitenStromSpeicher(?:_\d+)?\.xml", n)], key=natural_key)
        plant_files = sorted([n for n in names if re.fullmatch(r"AnlagenStromSpeicher(?:_\d+)?\.xml", n)], key=natural_key)
        if not unit_files or not plant_files:
            raise ValueError("The ZIP must contain both storage-unit and storage-plant XML files")
        batch = []
        for element in records(archive, unit_files, "EinheitStromSpeicher", source_members):
            counters["unit_records_seen"] += 1
            text = element.findtext
            if text("Land") != germany:
                counters["units_excluded_country"] += 1
                if (text("EinheitSystemstatus") == activated
                        and text("EinheitBetriebsstatus") == operating
                        and text("SpeMastrNummer")
                        and (not text("Inbetriebnahmedatum")
                             or text("Inbetriebnahmedatum") <= args.snapshot_date.isoformat())):
                    foreign_plant_links.add(text("SpeMastrNummer"))
                continue
            if text("EinheitSystemstatus") != activated:
                counters["units_excluded_system_status"] += 1
                continue
            if text("EinheitBetriebsstatus") != operating:
                counters["units_excluded_operating_status"] += 1
                continue
            commissioned = text("Inbetriebnahmedatum")
            if commissioned and commissioned > args.snapshot_date.isoformat():
                counters["units_excluded_future_commissioning"] += 1
                continue
            state = text("Bundesland") or ""
            if state not in states:
                counters["operating_german_units_without_valid_state"] += 1
                # Keep these in the join so a multi-state/missing-state plant
                # cannot have its entire energy assigned to a known state.
                state = "UNALLOCATED"
            plant = text("SpeMastrNummer") or ""
            technology = text("Technologie") or "UNKNOWN"
            net = number(text("Nettonennleistung"))
            gross = number(text("Bruttoleistung"))
            counters["eligible_units"] += 1
            counters["units_missing_or_invalid_net_power"] += net is None
            counters["units_missing_or_invalid_gross_power"] += gross is None
            counters["units_without_plant_link"] += not plant
            counters["units_without_commissioning_date"] += not commissioned
            batch.append((text("EinheitMastrNummer"), plant, state, technology, net, gross))
            if len(batch) >= 10000:
                db.executemany("INSERT INTO units VALUES(?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            db.executemany("INSERT INTO units VALUES(?,?,?,?,?,?)", batch)
        db.commit()
        db.execute("CREATE INDEX units_plant ON units(plant)")
        batch = []
        for element in records(archive, plant_files, "AnlageStromSpeicher", source_members):
            counters["plant_records_seen"] += 1
            batch.append((element.findtext("MaStRNummer"),
                          number(element.findtext("NutzbareSpeicherkapazitaet")),
                          element.findtext("AnlageBetriebsstatus")))
            if len(batch) >= 10000:
                db.executemany("INSERT INTO plants VALUES(?,?,?)", batch)
                batch.clear()
        if batch:
            db.executemany("INSERT INTO plants VALUES(?,?,?)", batch)
        db.commit()

    db.executescript("""
        CREATE TABLE plant_locations AS
        SELECT plant, MIN(state) AS state, MIN(technology) AS technology,
               COUNT(DISTINCT state) AS n_states,
               COUNT(DISTINCT technology) AS n_technologies,
               COUNT(*) AS operating_units, SUM(net_kw) AS net_kw
        FROM units WHERE plant != '' GROUP BY plant;
        CREATE UNIQUE INDEX plant_locations_id ON plant_locations(plant);
    """)
    # A shared foreign unit prevents assigning the entire plant energy to Germany.
    db.executemany("UPDATE plant_locations SET n_states=n_states+1 WHERE plant=?",
                   [(plant,) for plant in foreign_plant_links])
    counters["operating_foreign_plant_links_seen"] = len(foreign_plant_links)
    db.commit()
    # A plant's usable energy is counted once, irrespective of inverter count.
    # No allocation is invented for plants spanning states or technologies.
    power = {(s, t): (n, nn, ng, net or 0., gross or 0.)
             for s, t, n, nn, ng, net, gross in db.execute("""
        SELECT state,technology,COUNT(*),COUNT(net_kw),COUNT(gross_kw),
               SUM(net_kw),SUM(gross_kw) FROM units GROUP BY state,technology
    """)}
    energy = {(s, t): (n, known, value or 0.)
              for s, t, n, known, value in db.execute("""
        SELECT l.state,l.technology,COUNT(*),COUNT(p.energy_kwh),SUM(p.energy_kwh)
        FROM plant_locations l JOIN plants p ON p.id=l.plant
        WHERE l.n_states=1 AND l.n_technologies=1 AND p.status=?
        GROUP BY l.state,l.technology
    """, (operating,))}
    counters["linked_plants_spanning_states"] = db.execute("SELECT COUNT(*) FROM plant_locations WHERE n_states>1").fetchone()[0]
    counters["linked_plants_spanning_technologies"] = db.execute("SELECT COUNT(*) FROM plant_locations WHERE n_technologies>1").fetchone()[0]
    counters["linked_plants_absent_from_plant_table"] = db.execute("SELECT COUNT(*) FROM plant_locations l LEFT JOIN plants p ON p.id=l.plant WHERE p.id IS NULL").fetchone()[0]
    counters["linked_plants_not_operating"] = db.execute("SELECT COUNT(*) FROM plant_locations l JOIN plants p ON p.id=l.plant WHERE p.status!=?", (operating,)).fetchone()[0]
    counters["unallocated_known_energy_kwh"] = db.execute("""
        SELECT COALESCE(SUM(p.energy_kwh),0) FROM plant_locations l JOIN plants p ON p.id=l.plant
        WHERE p.status=? AND (l.n_states!=1 OR l.n_technologies!=1 OR l.state='UNALLOCATED')
    """, (operating,)).fetchone()[0]
    missing_links = {(s,t): n for s,t,n in db.execute("""
        SELECT u.state,u.technology,COUNT(*) FROM units u LEFT JOIN plants p ON p.id=u.plant
        WHERE p.id IS NULL GROUP BY u.state,u.technology
    """)}
    uncovered_energy = {(s,t): n for s,t,n in db.execute("""
        SELECT u.state,u.technology,COUNT(*) FROM units u
        LEFT JOIN plants p ON p.id=u.plant
        LEFT JOIN plant_locations l ON l.plant=u.plant
        WHERE p.id IS NULL OR p.status!=? OR p.energy_kwh IS NULL
              OR l.n_states!=1 OR l.n_technologies!=1
        GROUP BY u.state,u.technology
    """, (operating,))}
    flags = []
    flagged_energy = Counter()
    flagged_plants = Counter()
    for identifier, state, technology, energy_kwh, net_kw in db.execute("""
        SELECT p.id,l.state,l.technology,p.energy_kwh,l.net_kw
        FROM plant_locations l JOIN plants p ON p.id=l.plant
        WHERE p.status=? AND l.n_states=1 AND l.n_technologies=1
          AND l.technology=? AND l.net_kw>0 AND p.energy_kwh/l.net_kw>?
        ORDER BY p.energy_kwh DESC
    """, (operating, battery, args.battery_duration_flag_hours)):
        if state not in states:
            continue
        flagged_energy[state] += energy_kwh
        flagged_plants[state] += 1
        flags.append({"plant_mastr_number": identifier,
                      "state_iso": STATE_ISO[states[state]], "state": states[state],
                      "technology": technologies[technology],
                      "registered_usable_energy_MWh": round(energy_kwh/1000, 6),
                      "linked_net_power_MW": round(net_kw/1000, 6),
                      "energy_to_power_hours": round(energy_kwh/net_kw, 6),
                      "flag": f"battery_duration_exceeds_{args.battery_duration_flag_hours:g}_hours"})
    counters["battery_plants_flagged_extreme_duration"] = len(flags)
    counters["battery_energy_kwh_flagged_extreme_duration"] = sum(flagged_energy.values())
    tech_ids = sorted(set(technologies) | {t for _, t in power}, key=natural_key)
    detailed = []
    for state_id, state in sorted(states.items(), key=lambda x: x[1]):
        for technology in tech_ids:
            n, nn, ng, net, gross = power.get((state_id, technology), (0, 0, 0, 0., 0.))
            plants, known, usable = energy.get((state_id, technology), (0, 0, 0.))
            detailed.append({
                "snapshot_date": args.snapshot_date.isoformat(),
                "state_iso": STATE_ISO[state], "state": state,
                "technology_code": technology,
                "technology": technologies.get(technology, "Unknown / unmapped"),
                "operating_units": n, "units_with_net_power": nn,
                "net_power_MW": round(net / 1000, 6),
                "units_with_gross_power": ng, "gross_power_MW": round(gross / 1000, 6),
                "allocated_operating_plants": plants,
                "plants_with_usable_energy": known,
                "plants_missing_usable_energy": plants - known,
                "registered_usable_energy_MWh": round(usable / 1000, 6),
                "units_without_matching_plant_record": missing_links.get((state_id, technology), 0),
                "units_with_missing_or_unallocated_plant_energy": uncovered_energy.get((state_id, technology), 0),
            })
    summary = []
    for state_id, state in sorted(states.items(), key=lambda x: x[1]):
        rows = [r for r in detailed if r["state"] == state]
        battery_row = next(r for r in rows if r["technology_code"] == battery)
        totals = lambda field: round(sum(r[field] for r in rows), 6)
        summary.append({
            "snapshot_date": args.snapshot_date.isoformat(),
            "state_iso": STATE_ISO[state], "state": state,
            "battery_operating_units": battery_row["operating_units"],
            "battery_net_power_MW": battery_row["net_power_MW"],
            "battery_gross_power_MW": battery_row["gross_power_MW"],
            "battery_registered_usable_energy_MWh": battery_row["registered_usable_energy_MWh"],
            "battery_energy_MWh_excluding_duration_flags": round(battery_row["registered_usable_energy_MWh"] - flagged_energy[state_id]/1000, 6),
            "battery_plants_flagged_extreme_duration": flagged_plants[state_id],
            "battery_plants_with_usable_energy": battery_row["plants_with_usable_energy"],
            "battery_plants_missing_usable_energy": battery_row["plants_missing_usable_energy"],
            "battery_units_missing_net_power": battery_row["operating_units"] - battery_row["units_with_net_power"],
            "battery_units_with_missing_or_unallocated_plant_energy": battery_row["units_with_missing_or_unallocated_plant_energy"],
            "all_storage_operating_units": sum(r["operating_units"] for r in rows),
            "all_storage_net_power_MW": totals("net_power_MW"),
            "all_storage_gross_power_MW": totals("gross_power_MW"),
            "all_storage_registered_usable_energy_MWh": totals("registered_usable_energy_MWh"),
            "all_storage_plants_with_usable_energy": sum(r["plants_with_usable_energy"] for r in rows),
            "all_storage_plants_missing_usable_energy": sum(r["plants_missing_usable_energy"] for r in rows),
            "all_storage_units_missing_net_power": sum(r["operating_units"] - r["units_with_net_power"] for r in rows),
            "all_storage_units_with_missing_or_unallocated_plant_energy": sum(r["units_with_missing_or_unallocated_plant_energy"] for r in rows),
        })
    assert len(summary) == 16 and len({r["state_iso"] for r in summary}) == 16
    state_units = sum(r["all_storage_operating_units"] for r in summary)
    assert state_units + counters["operating_german_units_without_valid_state"] == counters["eligible_units"]
    write_csv(args.output_dir / "storage_by_state.csv", summary)
    write_csv(args.output_dir / "storage_by_state_and_technology.csv", detailed)
    write_csv(args.output_dir / "quality_checks.csv", [{"check": k, "value": v} for k, v in sorted(counters.items())])
    flag_fields = ["plant_mastr_number", "state_iso", "state", "technology",
                   "registered_usable_energy_MWh", "linked_net_power_MW",
                   "energy_to_power_hours", "flag"]
    with (args.output_dir / "battery_duration_flags.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flag_fields)
        writer.writeheader()
        writer.writerows(flags)
    metadata = {
        "source": "Bundesnetzagentur, Marktstammdatenregister (MaStR), public Gesamtdatenexport",
        "snapshot_date": args.snapshot_date.isoformat(), "source_url": args.source_url,
        "portal_url": "https://www.marktstammdatenregister.de/MaStR/Datendownload",
        "license": "Datenlizenz Deutschland - Namensnennung - Version 2.0",
        "license_url": "https://www.govdata.de/dl-de/by-2-0",
        "filters": {"Land": {germany: catalog[germany]},
                    "EinheitSystemstatus": {activated: catalog[activated]},
                    "EinheitBetriebsstatus": {operating: catalog[operating]},
                    "AnlageBetriebsstatus_for_energy": {operating: catalog[operating]},
                    "future_commissioning_dates_excluded": True},
        "field_units": {"Nettonennleistung": "kW", "Bruttoleistung": "kW",
                        "NutzbareSpeicherkapazitaet": "kWh"},
        "energy_join": "units.SpeMastrNummer = plants.MaStRNummer; usable energy counted once per plant",
        "missing_values": "Missing/negative/non-finite capacities are not imputed; reported energy is the sum of available nonnegative registered values. Consult coverage counts.",
        "allocation": "Energy of plants with multiple eligible states or technologies is unallocated. Units without a valid German state are excluded from state tables and recorded in quality_checks.csv.",
        "geography": "Physical location in the 16 German states; foreign-located plants and offshore economic zones are excluded.",
        "interpretation": "Registered operating stock at the snapshot, not dispatchable power, available energy, reserve eligibility, or a historical reconstruction.",
        "state_mapping": {k: {"name": v, "iso": STATE_ISO[v]} for k, v in states.items()},
        "technology_mapping": technologies,
        "battery_duration_screen": {
            "threshold_hours": args.battery_duration_flag_hours,
            "definition": "registered usable plant energy divided by summed eligible net unit power",
            "meaning": "Diagnostic flag, not a verified correction. Raw energy totals are retained. The separate screened battery-energy column excludes flagged energy without substituting a value. An anomaly may arise in either energy or power.",
            "flagged_plants": flags},
        "quality_checks": dict(counters),
        "totals_16_states": {
            field: round(sum(r[field] for r in summary), 6)
            for field in summary[0] if field not in {"snapshot_date", "state_iso", "state"}},
        "source_members": source_members,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["totals_16_states"], indent=2), flush=True)
    print(f"Completed in {time.monotonic()-start:.1f} seconds", flush=True)
    db.close()
    if temp:
        temp.cleanup()


if __name__ == "__main__":
    main()
