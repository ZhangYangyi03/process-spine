
# Provenance

Every table here is a real measurement record, redistributed under its upstream
licence, with the SHA-256 pinned and verified on load.

## buchwald/data_table.csv
- source: https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv
- upstream: https://github.com/doylelab/rxnpredict (MIT), from Ahneman et al., Science 2018
- sha256: fe310b50897e97578078558909efc2edaa7b98ad8939b1caad740a718398ed62
- 4599 rows, 0.998 of the 3x4x16x24 = 4608-cell product grid

## ccpp/ccpp.csv
- source: https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip
- upstream: UCI ML Repository #294, Kaya & Tufekci, 2012 (CC BY 4.0)
- sha256: 79e1c4524022de9468fc84289ab2c8f9ca5030790fa25e79e13443ddce4a24f3
- 9568 hourly records, 2006-2011; four ambient variables -> net hourly output
- the canonical xlsx was read with Excel COM (the host has no openpyxl/LibreOffice)
  and the four factor columns plus PE were written out; nothing else was changed

## gasturbine/gasturbine.csv
- source: https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip
- upstream: UCI ML Repository #551 (CC BY 4.0)
- 36733 records from gt_2011..gt_2015 concatenated; the target used is NOX
- sha256: 8d7e812a5e964bba5e09d3e2de9baab0a494e7b60bd11ef3b64ddf9e39874203
- rows with NOX == 0 dropped (they are instrument gaps, not clean combustion)

## concrete/concrete.csv
- source: https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip
- upstream: UCI ML Repository #165, I-Cheng Yeh, 2007 (CC BY 4.0)
- sha256: 98072ff035fc27116079be5c0075c3144677c66877fb5d95a026f1404af80aa5
- 1030 mix designs; eight mix variables and curing age -> compressive strength

## The quantisation is declared, not hidden
For the three continuous tables each factor is cut into n equal-width levels
across [q01, q99] (buchwald is untouched: its factors are already named). Rows
snap to the nearest grid point. `meta["snap_max_frac_of_range"]` gives half a
grid step per factor as a fraction of that factor's range, and the loader prints
it. The surrogate reads the grid index as a *scaled metric coordinate*, so the
gradient survives the quantisation; what does not survive is resolution finer
than one step.
