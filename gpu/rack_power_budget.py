# rack_power_budget.py
# Companion example for: topnotchnote.com/gpu/rack_power_cooling.html
#
# Back-of-envelope power arithmetic for GPU systems, from vendor figures:
#   * kW per GPU, all-in (GPUs + CPUs + NICs + fans), for each system
#   * how many such GPUs a "typical" rack's power budget can feed
#   * facility power for a cluster, once cooling and power delivery overhead
#     are added through PUE (total facility power / IT power)
# Edit the SYSTEMS list or pass a GPU count to try your own cluster.
#
# Sources for the defaults are cited on the page:
#   DGX H100 heat output 38,557 BTU/hr (Nvidia DGX H100 user guide)
#   MI350X air-cooled: 8 servers (64 GPUs) in a 120-130 kW rack,
#   MI355X liquid-cooled: 128 GPUs in a 200 kW rack (AMD CDNA 4 white paper)
#   typical rack 7.5 kW and weighted average PUE 1.54 (Uptime Institute 2025)
#
# Run:  python rack_power_budget.py [gpus_in_cluster]

import sys

BTU_PER_HOUR_PER_WATT = 3.412        # unit conversion
TYPICAL_RACK_KW = 7.5                # Uptime 2025, modal density excl. outliers
PUE = 1.54                           # Uptime 2025 weighted average

SYSTEMS = [
    # name, GPUs, kW for that many GPUs. The DGX figure is the server's rated
    # heat output; the AMD figures are the rack power budget AMD says those
    # GPUs fit in, an upper bound rather than a measured draw.
    ("DGX H100 server (air), heat output", 8, 38557 / BTU_PER_HOUR_PER_WATT / 1000),
    ("MI350X rack (air), 125 kW budget", 64, 125.0),
    ("MI355X rack (liquid), 200 kW budget", 128, 200.0),
]

gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 16384

print(f"{'system':36s} {'kW':>7s} {'kW/GPU':>7s} {'GPUs per 7.5 kW rack':>21s}")
for name, n, kw in SYSTEMS:
    per = kw / n
    print(f"{name:36s} {kw:7.1f} {per:7.2f} {TYPICAL_RACK_KW / per:21.1f}")

print(f"\ncluster of {gpus:,} GPUs, facility PUE {PUE}:")
print(f"{'system':36s} {'IT MW':>7s} {'facility MW':>12s} {'typical racks':>14s}")
for name, n, kw in SYSTEMS:
    it_mw = gpus / n * kw / 1000
    print(f"{name:36s} {it_mw:7.1f} {it_mw * PUE:12.1f} {it_mw * 1000 / TYPICAL_RACK_KW:14,.0f}")
print("\n'typical racks' = how many 7.5 kW racks the same IT load would fill")
