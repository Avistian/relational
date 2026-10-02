# L175 measured audit

| Check | Complete measured result |
|---|---:|
| Contexts / cell slots | 2,106 / 2,156,544 |
| Independently reconstructed query labels | 2,106 |
| Exposed query targets | 0 |
| Same-time visible outcome labels | 0 |
| Visible labels with unfinished outcome windows | 0 |
| Future-dated cells / affected contexts | **385 / 77** |
| Visible historical label cells | 29,988 |
| Historical test-period label cells | 12,351 |
| Cells without a timestamp | 432,050 |
| Checkpoint evaluations | **0 of 6 — NOT_RUN** |

Counts include repeated cells across query contexts. They are not counts of unique database rows. Context seeds0/1/2 contain26/26/25affected queries respectively.

All future-dated cells belong to **race schedule rows**: year, round, name, date and time. One query at **2011-03-27 00:00 UTC** includes a race dated **06:00 UTC** that day. The sampler reaches it through an unfiltered foreign-key-to-parent step. These attributes may have been known in advance; the snapshot contains no arrival history to prove that. We establish failure of the declared event-time filter, **not demonstrated exposure to future race outcomes**. The complete audit found no unmasked query target and no label with an unfinished outcome window.
