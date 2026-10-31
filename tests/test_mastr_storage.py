"""Checks for the consequential MaStR aggregation edge cases (no network)."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "aggregate_mastr_storage.py"
spec = importlib.util.spec_from_file_location("mastr_storage", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def xml(root_name, record_name, records):
    root = ET.Element(root_name)
    for record in records:
        item = ET.SubElement(root, record_name)
        for key, value in record.items():
            if value is not None:
                ET.SubElement(item, key).text = str(value)
    return ET.tostring(root, encoding="utf-16", xml_declaration=True)


class AggregationTest(unittest.TestCase):
    def test_shared_plants_missing_energy_exclusions_and_no_invented_allocation(self):
        state_names = list(module.STATE_ISO)
        state_ids = {name: str(1400 + i) for i, name in enumerate(state_names)}
        categories = [("84", "Deutschland", "6"), ("90", "Dänemark", "6"),
                      ("35", "In Betrieb", "4"), ("31", "In Planung", "4"),
                      ("472", "Aktiviert", "19"), ("473", "Deaktiviert", "19"),
                      ("524", "Batterie", "29"), ("1537", "Pumpspeicher", "29")]
        categories += [(identifier, name, "101") for name, identifier in state_ids.items()]
        catalog = [{"Id": i, "Wert": name, "KatalogKategorieId": category}
                   for i, name, category in categories]
        units = []

        def unit(identifier, plant, state="Bayern", net="1000", **overrides):
            row = {"EinheitMastrNummer": identifier, "SpeMastrNummer": plant,
                   "Land": "84", "Bundesland": state_ids[state],
                   "EinheitSystemstatus": "472", "EinheitBetriebsstatus": "35",
                   "Technologie": "524", "Nettonennleistung": net,
                   "Bruttoleistung": "1200", "Inbetriebnahmedatum": "2024-01-01"}
            row.update(overrides)
            units.append(row)

        unit("U1", "P1")
        unit("U2", "P1")  # Two inverters: P1's 5000 kWh must count once.
        unit("U3", "P2", net=None)  # Known unit with missing power and energy.
        unit("U4", "P3")
        unit("U5", "P3", state="Berlin")  # No state allocation for shared energy.
        unit("U6", "P4", Land="90")
        unit("U7", "P4", EinheitBetriebsstatus="31")
        unit("U8", "P4", EinheitSystemstatus="473")
        unit("U9", "P4", Inbetriebnahmedatum="2027-01-01")
        unit("U10", "P5", net="0")  # Zero is distinct from missing.
        unit("U11", "P6")
        unit("U12", "P6", Land="90")  # Shared cross-border energy is unallocated.
        unit("U13", "P7", net="1")  # Extreme duration: retain raw, expose screened value.
        plants = [{"MaStRNummer": name, "AnlageBetriebsstatus": "35",
                   "NutzbareSpeicherkapazitaet": value}
                  for name, value in [("P1", "5000"), ("P2", None), ("P3", "10000"),
                                      ("P4", "1000"), ("P5", "0"), ("P6", "500"),
                                      ("P7", "100000")]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("Katalogwerte.xml", xml("Katalogwerte", "Katalogwert", catalog))
                archive.writestr("EinheitenStromSpeicher_1.xml", xml("EinheitenStromSpeicher", "EinheitStromSpeicher", units))
                archive.writestr("AnlagenStromSpeicher_1.xml", xml("AnlagenStromSpeicher", "AnlageStromSpeicher", plants))
            subprocess.run([sys.executable, str(SCRIPT), str(archive_path),
                            "--snapshot-date", "2026-10-31", "--output-dir", str(root / "output")],
                           check=True, capture_output=True, text=True)
            with (root / "output" / "storage_by_state.csv").open() as handle:
                rows = {r["state"]: r for r in csv.DictReader(handle)}
            self.assertEqual(len(rows), 16)
            self.assertEqual(int(rows["Bayern"]["battery_operating_units"]), 7)
            self.assertEqual(float(rows["Bayern"]["battery_net_power_MW"]), 4.001)
            self.assertEqual(float(rows["Bayern"]["battery_registered_usable_energy_MWh"]), 105.0)
            self.assertEqual(float(rows["Bayern"]["battery_energy_MWh_excluding_duration_flags"]), 5.0)
            self.assertEqual(int(rows["Bayern"]["battery_plants_flagged_extreme_duration"]), 1)
            self.assertEqual(int(rows["Bayern"]["battery_plants_missing_usable_energy"]), 1)
            self.assertEqual(int(rows["Bayern"]["battery_plants_with_usable_energy"]), 3)
            self.assertEqual(float(rows["Berlin"]["battery_registered_usable_energy_MWh"]), 0.0)
            metadata = json.loads((root / "output" / "provenance.json").read_text())
            checks = metadata["quality_checks"]
            self.assertEqual(checks["linked_plants_spanning_states"], 2)
            self.assertEqual(checks["unallocated_known_energy_kwh"], 10500.0)
            self.assertEqual(checks["units_excluded_future_commissioning"], 1)
            self.assertEqual(checks["units_missing_or_invalid_net_power"], 1)


if __name__ == "__main__":
    unittest.main()
