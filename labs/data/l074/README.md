# L074 source data and representation cache

`manifest.json` records exact upstream row IDs and input hashes. Parquet files are label-blind 384-row subsets of the pinned CARTE examples, with exact entity-name duplicates removed before sampling. Three tables are evaluated independently.

`vectors.npz` contains 2,713 real 300-dimensional sentence vectors generated with the FastText English crawl binary; includes column names and observed strings, not target values. Cache preparation uses fixed pretrained parameters and fits no target-table statistics. String values are lowercased, as in the CARTE release; column names retain their exact spelling.

FastText vectors: https://fasttext.cc/docs/en/crawl-vectors.html ; CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/ . Credit to the FastText authors. This derived cache retains those terms.

`kg_pretrained.pt` is the unmodified YAGO-pretrained CARTE checkpoint from commit f54690da4cddbedd1e1a9113a312f85783d2c125. Only the local encoder's strict matching initial maps and final readout tensors are used. The full checkpoint is retained for auditability. CARTE source license: ../../sources/carte-l074/LICENSE.txt.

See ../../l074-reproduction.md for leakage boundaries, target transformations and regeneration commands. Rebuilding the cache requires a large language-model binary once; the default student lab does not.
