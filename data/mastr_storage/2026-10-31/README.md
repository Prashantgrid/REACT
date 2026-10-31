# Registered electricity storage by German federal state

One Bundesnetzagentur Marktstammdatenregister (MaStR) public XML snapshot, dated
**31 October 2026**. All state totals come from that same export.

## Files

- `storage_by_state.csv`: the main table, exactly 16 state rows, with battery-specific
  and all-storage totals for registered net power (MW), gross power (MW), and usable
  energy (MWh), plus counts that expose missing data.
- `storage_by_state_and_technology.csv`: the same data separated by the storage
  technologies in the snapshot catalogue. Zero-count technology/state combinations
  remain in the table.
- `quality_checks.csv`: source-record counts, exclusions, missing values, unmatched
  links, and unallocated energy.
- `battery_duration_flags.csv`: individually auditable battery records whose registered
  energy/net-power ratio exceeds 8,760 hours (one year at registered net power).
- `provenance.json`: source URL, filters, catalogue mappings, totals and SHA-256/CRC
  fingerprints of the XML members used.

CSV files use UTF-8, comma delimiters and decimal points. Preserve German state
names or join with the `state_iso` column. MW and MWh are different quantities.

## Official source and attribution

Provider: **Bundesnetzagentur, Marktstammdatenregister (MaStR)**.

- [Download portal](https://www.marktstammdatenregister.de/MaStR/Datendownload)
- [Snapshot used](https://download.marktstammdatenregister.de/Gesamtdatenexport_20261031_26.2.zip)
- [Export documentation](https://www.marktstammdatenregister.de/MaStRHilfe/files/gesamtdatenexport/Dokumentation%20MaStR%20Gesamtdatenexport.zip)
- License: [Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0).

Processing: German-state aggregation, operating-status filtering and unit conversion
from the original public registry fields. This is a derived dataset, not an official
Bundesnetzagentur state-level statistical release. Daily snapshot URLs may subsequently
be withdrawn; retain the original archive when reproducing this exact snapshot. The
provider's dated quarterly exports are an alternative when building a new snapshot.

## Selection and aggregation

The script reads `Katalogwerte.xml`, all `EinheitenStromSpeicher_*.xml`, and all
`AnlagenStromSpeicher_*.xml` from the same ZIP. All required members are read completely,
with ZIP CRC verification and uncompressed SHA-256 fingerprints. Other generation types
and unrelated registry data are not inputs.

1. Select physically German-located units (`Land=Deutschland`) with an activated
   system record and operating status `In Betrieb`. Exclude future commissioning
   dates. Retain units whose commissioning date is missing and report their count.
2. Map the registered `Bundesland` to the 16 federal states using the snapshot's
   own catalogue. Offshore economic zones, foreign locations and unknown states
   are not assigned to a German state. Relevant exclusions are reported.
3. Sum unit `Nettonennleistung` and `Bruttoleistung`, each in kW, then divide by
   1,000 to obtain MW. Keep net and gross power distinct; missing net power is
   not replaced by gross power.
4. Join `EinheitenStromSpeicher.SpeMastrNummer` to
   `AnlagenStromSpeicher.MaStRNummer`. Sum the plant's registered
   `NutzbareSpeicherkapazitaet` in kWh once per plant, then divide by 1,000 for MWh.
   A multi-inverter plant's energy is therefore counted once.
5. Include a plant's energy only when its plant status is `In Betrieb` and its
   eligible linked units identify one state and one technology, with no shared
   operating foreign unit. Do not invent an
   energy allocation for plants spanning states or technologies. Such energy is
   recorded as unallocated in the checks.
6. Report batteries separately (`Technologie=Batterie`); all-storage totals also
   include the other registered technologies where present. All plant sizes and
   battery chemistries are included, including household/behind-the-meter systems.

Missing, negative or non-finite capacity values are not imputed. Zero remains a
registered zero. Energy sums cover available nonnegative registered values; inspect
the accompanying counts before treating them as complete totals. A zero energy sum
can therefore coexist with units whose energy is missing or unallocated. Counts of
plants and units differ because a plant can have multiple units.

## Material source-quality anomaly

**The raw register contains an extreme battery-energy entry:** plant
`SSE913520479976` in Nordrhein-Westfalen reports 30,042.026 MWh alongside 0.020 MW
of linked net power. Its energy/power ratio is about 1.50 million hours. This single
entry materially inflates the raw state and national energy sums.

The main table retains `battery_registered_usable_energy_MWh` unchanged and also
provides `battery_energy_MWh_excluding_duration_flags`. The latter excludes the
energy of battery plants whose energy/net-power ratio exceeds the explicitly stated
8,760-hour diagnostic threshold. The raw power sum is retained. This is a sensitivity
comparison, not a verified corrected dataset; the error could be in the energy or
power field. No replacement capacity is estimated. The threshold is configurable
with `--battery-duration-flag-hours` and is recorded in `provenance.json`.

Other technology totals, including pumped storage, remain raw registry aggregates.
Neither a duration flag nor its absence establishes the validity of a registration.

## Interpretation for REACT

These tables describe **registered operating stock on 31 October 2026**. They do
not estimate actual state of charge, dispatch availability, firm capacity, controllability,
or eligibility to provide reserves. In particular, a household battery's full registered
capacity is not automatically available to an interstate support pool.

Usable energy is taken directly from the registered energy field; no four-hour
power-to-energy conversion or PV-proportional allocation is used. These are registry
values, not independently measured or comprehensively network-operator-verified
capacities. Raw extreme values are not silently capped; inspect unusual values before
using the tables in a model. Registration delays and later corrections can affect totals.

This 2026 snapshot is used only as a later-date sensitivity. The revised manuscript's
main battery input uses the archived 1 October 2025 export. Commissioning dates in
a later register do not reconstruct the complete historical stock.

## Reproduce

Download and retain the official archive through the MaStR portal, then run from the
repository root:

```bash
pip install -r requirements-storage.txt
python tools/aggregate_mastr_storage.py /path/to/Gesamtdatenexport_20261031_26.2.zip \
  --snapshot-date 2026-10-31 \
  --source-url https://download.marktstammdatenregister.de/Gesamtdatenexport_20261031_26.2.zip \
  --output-dir data/mastr_storage/2026-10-31
python -m unittest discover -s tests -p test_mastr_storage.py -v
```

The script streams the XML and uses a temporary SQLite database for the join. The
network-free test checks shared-inverter deduplication, missing versus zero values,
status/country/date exclusions, and the absence of invented cross-state or cross-border
energy allocation.

The repository stores the derived aggregate tables and reproducibility code. The large
original ZIP and temporary database are not committed.
