# Year 5 entry: bounded source currency check

2026-10-01. Scope: verify the introductory source and look for consequential changes at the quarter boundary; no training or new ranking claims. Web search fallback used; no arXiv MCP is available.

- Bommasani v1 (2021-08-16) remains the conceptual source. The latest arXiv landing page points to v3 (2022-07-12); cite the chosen version explicitly.
- Verified https://arxiv.org/abs/2609.00460v1 (Context Window Failures), already core B18 in the bridge. Its stated synthetic high-cardinality failure motivates explicit context limits; it is not a universal verdict about RFMs.
- Verified https://arxiv.org/abs/2608.16319v2 (RelArena-α/TabPFN-Rel/RPI), already B14. It strengthens the need for comparable protocols and flattening baselines; do not borrow its ranking as a current independently checked leaderboard position.
- Verified https://arxiv.org/abs/2609.37959 (TabFM). The bridge already names TabFM in its candidate inventory. Full method/code and contamination review remains pending before adding a required reproduction. L161 is unchanged.
- RelBench's root redirects to https://star-project.stanford.edu/relbench/. The old /leaderboard.html request failed. TabArena redirects to its Hugging Face leaderboard space; extracted HTML did not expose the ranking. Current leaderboard positions NOT_VERIFIED; no SOTA claim is made.

No additions satisfy a completed method/protocol audit in this bounded pass. Keep the existing sequence and use its full currency process before later model-specific lessons. These checks establish source identities and scope, not fresh benchmark results or exhaustive search coverage.
