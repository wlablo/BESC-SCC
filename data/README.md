# Data sources: download separately

Original third-party datasets are **not redistributed** in this repository.
For replication, place the exact following filenames here:

- `email-Eu-core.txt.gz` — [Stanford SNAP email-Eu-core](https://snap.stanford.edu/data/email-Eu-core.html)
- `email-Eu-core-department-labels.txt.gz` — same page
- `CollegeMsg.txt.gz` — [Stanford SNAP CollegeMsg](https://snap.stanford.edu/data/CollegeMsg.html)
- `polblogs.gml` — [Political blogs network, Newman collection](https://public.websites.umich.edu/~mejn/netdata/). Download the **GML** representation with vertex `value` (group labels). Some mirrors provide only a different representation; the loader currently expects the GML format.

Use the sources under their providers' applicable terms and cite the original publications.
Do not substitute the temporal email-Eu-core version for the static email-Eu-core dataset.

Expected data *after preprocessing*:

| Dataset | Nodes | Directed arcs after removing self-loops | Groups |
|---|---:|---:|---:|
| email-Eu-core | 1005 | 24929 | 42 observed departments |
| Polblogs | 1490 | 19014 | 2 observed categories |
| CollegeMsg | 1899 | 20296 | 16 inferred Louvain groups with seed 17 |

The CollegeMsg input consists of 59835 temporal message events that are collapsed to unique static arcs, and the community labels are computed (not observed). These groups depend on the software version; save the exact environment and results manifest when reproducing.

Run `python scripts/check_inputs.py` after placing the input files.
