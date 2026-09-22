# Lower-cost Azure GPU candidates

Checked 2026-09-21 with the Azure Retail Prices API, official size specifications
and the deployment subscription's SKU/quota metadata. USD prices below are Linux
VM compute only, excluding disks, NAT, Private Link, taxes and negotiated discounts.
Spot prices vary and do not guarantee available capacity.

| Candidate, West US 2 | RAM / VRAM | Regular USD/hour | Spot USD/hour | Assessment |
|---|---|---:|---:|---|
| `Standard_NV36ads_A10_v5` | 440 GiB / 24 GB | 3.20 | 0.591360 | Previous cost baseline |
| `Standard_NC24lds_xl_RTXPRO6000BSE_v6` | 72 GiB / 24 GB, 1/4 Blackwell GPU | 1.13 | 0.208824 | Preferred lower-cost **benchmark candidate**, not yet validated on fractional GPU |
| `Standard_NC8as_T4_v3` | 56 GiB / 16 GB | 0.752 | 0.213267 | Cheaper regular pricing, but older GPU and explicit graphics-driver requirements |
| `Standard_NC4as_T4_v3` | 28 GiB / 16 GB | 0.526 | 0.149174 | Excluded from the baseline: below Isaac Sim 5.1's 32 GB system-memory minimum |

The smaller Blackwell size costs about **65% less at regular pricing** than the
previous A10 size; its listed Spot rate is about 93% lower than that regular A10
baseline. This is a price comparison, not a measured TCO or performance result.
Legacy `Low Priority` price records were excluded: they are not interchangeable
with current Spot offers.

## Compatibility and actual capacity are separate

Isaac Sim 5.1 publishes a 32 GB RAM / 16 GB VRAM minimum and an RTX 4080 minimum
GPU reference. RT cores are required; A100/H100 are not supported for this
renderer. Merely meeting memory numbers is not proof of version/driver support.

The 24 GB Blackwell slice meets those memory numbers and the Azure family is
intended for Omniverse workloads, but the **exact fractional slice, driver,
renderer, scene and camera load must pass the actual compatibility checker**.
The T4 is below the currently published minimum GPU performance reference.
Microsoft also says NCasT4 graphics workloads require the supported GRID driver;
the compute-oriented driver extension alone is not an established graphics setup.

Current subscription observations:

- West US 2 lists the small Blackwell SKU without a SKU-level restriction.
- Standard Blackwell and T4 family quotas remain zero.
- T4 SKUs in the checked regions report `NotAvailableForSubscription`.
- A **separate Spot quota pool** reports 100 unused vCPUs. That does not itself
  establish the offer's Spot eligibility or successful allocation.
- The subscription reports an internal offer. Do not infer its Spot eligibility
  solely from the generic list of public supported offers.
- Azure Resource Graph returned no applicable Spot-history rows for this query;
  the numbers above therefore come from the retail price API, not an eviction estimate.

No GPU was created or NVIDIA agreement accepted during this comparison.
For a customer presentation, prefer regular capacity after validation; Spot is
appropriate only when interruption is acceptable. Spot has no availability SLA
and can be reclaimed with 30 seconds notice. A maximum compute price does not cap
disk/network charges, and a deallocated Spot VM can still incur disk charges.

## Subsequent actual allocation probe

On 2026-09-21 a real `Standard_NC24lds_xl_RTXPRO6000BSE_v6` **Spot** VM in
West US 2 was successfully created using `scripts.gpu_capacity_probe`.
Its instance view reported `PowerState/running`, and a guest startup probe read
an NVIDIA display-class PCI device (`vendor=0x10de`, `device=0x2bb5`) from sysfs.
This was an actual allocation, not merely a price or quota lookup.

The VM compute price ceiling was USD 0.30/hour. It was explicitly deallocated
after verification; a one-hour UTC shutdown schedule was also configured.
There is no public IP and inbound traffic is denied. Failed probes may clean
only resources carrying their exact random probe ID.

The probe did not install NVIDIA applications, accept NVIDIA terms, run Isaac
Sim, or prove graphics-driver/rendering compatibility. Those remain separate
gates. The retained deallocated VM/disk can be inspected by the operator;
restarting Spot capacity is not guaranteed and retained disks incur charges.
The regular-family quota rejection did not imply that the separate Spot quota
pool could not allocate this VM.

## Subsequent actual runtime and interruption

After the operator authorized runtime installation and prioritized correctness
over price, the smaller Spot allocation could not be restarted. The retained
installation was resized to `Standard_NC72lds_xl_RTXPRO6000BSE_v6` and actually ran
with a 48 GB RTX PRO Blackwell vGPU, GRID 595.91.07 and Isaac Sim 6.0.0.
Real rendering and physical grasp/sort/retreat were observed. Isaac 5.1 failed
renderer initialization on this vGPU; the older memory comparison is not a
compatibility claim or the currently deployed SKU.

Azure recorded `evictSpotVM` at 2026-09-21 14:05:57 UTC and deallocated the VM.
It was subsequently restarted, but recovery does not remove interruption risk.
Setting maximum Spot price to `-1` avoids price-based eviction, not capacity eviction.

Fresh regular RTX/A10 family quota checks in West US 2, East US 2, West US 3 and
East US still returned zero. A West US 3 regular A10 request for 36 vCPUs returned
`ContactSupport`. Regular capacity therefore remains an external deployment
prerequisite for an uninterrupted customer presentation; do not describe the
current Spot runtime as a dependable always-on service.

The recovery then allocated a separate private `Standard_NV36ads_A10_v5` Spot
VM rather than forcing the validated Blackwell VM's NVMe boot disk onto a SCSI
size. It inherited the protected subnet and simulator identity, without a public
IP. Microsoft's A10-specific GRID 570.237 driver, Vulkan `NVIDIA A10-24Q`, the
unchanged Isaac 6 image and actual normal/defect physical sorting were verified.
The image declares `MIN_DRIVER_VERSION=570.169`; no driver-check bypass was used.
The old Blackwell VM remains deallocated with its disk intact.

A read-only audit of an actual A10 presentation verified 48 distinct successful
cycles, original Blob input images, model response IDs, scene epochs, correct
destinations and 12.75-18.98 second physical durations. A later prolonged reset
sequence exposed RTX descriptor exhaustion; the reference renderer now reuses
identical-geometry resources, with separate actual validation of that correction.
This history is not a Spot availability guarantee or a learned-policy benchmark.

## Sources

- [Azure Retail Prices API](https://prices.azure.com/api/retail/prices)
- [NC RTX PRO 6000 BSE v6 specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nc-rtxpro6000-bse-v6-series)
- [NCasT4 v3 specifications and driver note](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series)
- [Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Azure Spot VM pricing, quota and eviction behavior](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms)
- [Azure Linux GRID driver matrix](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
