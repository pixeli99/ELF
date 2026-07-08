# Checkpoint Manifest

Large checkpoints are kept locally and are not committed to GitHub.

| model | checkpoint path | note |
|---|---|---|
| ordered | outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592 | 5B-token continuation checkpoint |
| register | outputs/elf_b-owt-register-5b-4g/checkpoint_152592 | 5B-token continuation checkpoint |
| vanilla | outputs/elf_b-owt-vanilla-5b-4g/checkpoint_152592 | 5B-token continuation checkpoint |
| ordered full old run | outputs/elf_b-owt-ordered/checkpoint_304287 | previous ordered checkpoint |
| smoke diagonal | outputs/smoke_diagonal_initfrom_1step/checkpoint_1 | smoke test checkpoint |
| smoke ordered | outputs/smoke_ordered_initfrom_1step/checkpoint_1 | smoke test checkpoint |
| smoke register | outputs/smoke_register_initfrom_1step/checkpoint_1 | smoke test checkpoint |
| smoke vanilla | outputs/smoke_vanilla_initfrom_1step/checkpoint_1 | smoke test checkpoint |

All 5B continuation runs use the same base ELF-B checkpoint and `max_optimizer_steps=9537`.
