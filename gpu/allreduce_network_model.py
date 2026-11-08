# allreduce_network_model.py
# Companion example for: topnotchnote.com/gpu/cluster_networking.html
#
# How long does one gradient all-reduce take on a given network?
# A ring all-reduce over n ranks moves 2(n-1)/n times the buffer through each
# rank's link (the "bus bandwidth" factor in NCCL's performance notes), so a
# lower bound on its time is
#     t = 2 (n - 1) / n * bytes / link_bandwidth
# where link_bandwidth is the slowest per-GPU link on the ring. Inside a
# server that is NVLink or Infinity Fabric; across servers it is the GPU's
# share of the network (one 400 Gb/s NIC per GPU on a DGX H100, or what is
# left of it after oversubscription).
#
# This is arithmetic, not a measurement: it ignores latency, protocol
# overhead and overlap with compute. Use it to see which link dominates and
# what a slower or oversubscribed network costs, then measure with
# nccl-tests / rccl-tests on the real cluster.
#
# Run:  python allreduce_network_model.py [params_in_billions] [gpus]

import sys

params_b = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0   # model size
gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 128          # ranks
grad_bytes = params_b * 1e9 * 2                                 # BF16 gradients
factor = 2 * (gpus - 1) / gpus

links = [
    # (name, GB/s per GPU, one direction)
    ("NVLink 4 (as if every hop were NVLink)", 450.0),
    ("InfiniBand or RoCE, 400 Gb/s per GPU", 400 / 8),
    ("same, behind 1:7 oversubscription", 400 / 8 / 7),
    ("Ethernet, 100 Gb/s per GPU", 100 / 8),
    ("Ethernet, 25 Gb/s per server shared by 8 GPUs", 25 / 8 / 8),
]

print(f"{params_b:g}B parameters, BF16 gradients = {grad_bytes / 1e9:.1f} GB, "
      f"ring all-reduce over {gpus} GPUs (factor {factor:.3f})\n")
print(f"{'slowest link on the ring':48s} {'GB/s':>8s} {'seconds':>9s}")
for name, gbps in links:
    t = factor * grad_bytes / (gbps * 1e9)
    print(f"{name:48s} {gbps:8.2f} {t:9.3f}")
print("\nlower bounds: real collectives add latency and protocol overhead")
