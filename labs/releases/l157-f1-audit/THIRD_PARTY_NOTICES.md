# Third-party notices and provenance

RelBench source files in labs/sources/l117 are MIT licensed; preserve their adjacent LICENSE. Pinned commit: 9aa346267c2e1c560bd92da07d6f4ad1ca2f0639. SQL comes from relbench-user-study commit445bb7a3b1230f49f8e5890ae81754d3e365680f; no license was found at its pinned root, so it is downloaded on demand and is not bundled or relicensed. The inherited Frame adapter notice is retained in labs/sources/l129/frame/LICENSE. Course implementations retain their provenance docstrings and are provided under the root MIT license. Dependencies keep their respective licenses.

Raw database/task archives and pretrained text weights are downloaded from their original hosts, never bundled or relicensed here. SHA256 and a text-model revision identify the expected artifacts. Availability of those upstream hosts is required for fresh runs. Code licensing does not establish permission to redistribute every upstream dataset. The compact evidence is derived benchmark output, not the raw database.

Primary paper: Robinson et al., RelBench: A Benchmark for Deep Learning on Relational Databases, NeurIPS2024, https://arxiv.org/abs/2407.20060. Released baseline target: Table7 basic GNN, rel-f1/driver-position. The fixed2005 intervention is course-specific.
