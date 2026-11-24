# DSFlash PSR implementation plan

The [design](../specs/2026-11-24-dsflash-psr.md) covers the network, data protocol, and run setup.

- [x] Add failing checks for mixed mask counts, overlap, exchange consistency, and multihot labels.
- [x] Extract the reusable Swin feature encoder with compatible KAF checkpoint keys, and add the DSFlash relation head.
- [x] Add the grouped split generator, independent-mask PSR loader, and ID-based relation metrics.
- [x] Add the configuration, torchrun training and checkpoint/evaluation entry points, and inference from supplied masks.
- [ ] Run local and labserver0 smoke checks, then deploy and verify the four-GPU launch after 14:00.
