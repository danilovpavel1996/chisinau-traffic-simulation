# Network QA Checklist — Airport ↔ Botanica Corridor

Fill this in as you verify the network in SUMO-GUI.
Confidence score reflects how well OSM matches reality for each item.

---

## 10 Key Intersections

| # | Intersection ID (SUMO) | Real-world label | Lanes verified | Speed verified | TLS present | Turn restrictions | Confidence | Notes |
|---|------------------------|------------------|:-:|:-:|:-:|:-:|:-:|---|
| 1 | J_AEROPORT | Aeroport Junction (main entrance) | ☐ | ☐ | ☐ | ☐ | — | |
| 2 | J_DACIA | Dacia Blvd × Str. Aeroportului | ☐ | ☐ | ☐ | ☐ | — | |
| 3 | J_MUNCII | Str. Muncii crossing | ☐ | ☐ | ☐ | ☐ | — | |
| 4 | J_INDEPENDENTEI | Calea Independentei × arterial | ☐ | ☐ | ☐ | ☐ | — | |
| 5 | J_BOTANICA_MAIN | Botanica main corridor entry | ☐ | ☐ | ☐ | ☐ | — | |
| 6 | J_CIUFLEA | Ciuflea Rd junction | ☐ | ☐ | ☐ | ☐ | — | |
| 7 | J_ISMAIL | Str. Ismail T-junction | ☐ | ☐ | ☐ | ☐ | — | |
| 8 | J_VASILE_LUPU | Vasile Lupu Blvd split | ☐ | ☐ | ☐ | ☐ | — | |
| 9 | J_GRENOBLE | Grenoble Rd roundabout | ☐ | ☐ | ☐ | ☐ | — | |
|10 | J_RISCANI_GATE | Riscani district gateway | ☐ | ☐ | ☐ | ☐ | — | |

## 2–3 Key Arterials

| Arterial | Direction | Lanes (expected / OSM / verified) | Speed limit (km/h) | One-way correct | Confidence | Notes |
|----------|-----------|:-:|:-:|:-:|:-:|---|
| Calea Iesilor / Str. Aeroportului | N–S | 2 / ? / ? | 60 | ☐ | — | Main airport access road |
| Dacia Blvd | E–W | 3 / ? / ? | 60 | ☐ | — | Primary east-west connector |
| Vasile Lupu Blvd | N–S | 2 / ? / ? | 50 | ☐ | — | Botanica spine |

---

## Known Issues / Patches Applied

| Date | Edge/Node ID | Issue | Fix applied | Confidence |
|------|-------------|-------|-------------|:-:|
| — | — | — | — | — |

---

## Overall Network Confidence: **0.85** (target before Milestone A3)

_Update this doc every time you make a structural change to build_sumo_net.py NETWORK_PATCHES._
