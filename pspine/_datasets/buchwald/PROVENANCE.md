# Provenance

`data_table.csv`
- source: https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv
- upstream repo: https://github.com/doylelab/rxnpredict (MIT)
- upstream commit date: 2018 (contents match the Science 2018 paper's SI table)
- SHA-256: fe310b50897e97578078558909efc2edaa7b98ad8939b1caad740a718398ed62
- fetched: 2026-10-08
- verified on every load by `pspine.data.buchwald()`; a mismatch raises.

`plate1.1_raw.csv`
- source: https://raw.githubusercontent.com/doylelab/rxnpredict/master/yield_data/plate1.1.csv
- one raw HPLC plate as the instrument produced it, kept so the tidy table can be
  traced back to a measurement rather than to a paper's summary.
- SHA-256: 13f722bd1d9491a82dcef004935884bdfed8dd2595c4bd5767866308fe0e59ea

`LICENSE_rxnpredict.txt`
- the upstream MIT licence, kept verbatim because the data is redistributed here.

The tidy table's four factor columns are exactly the levels the experimenters
varied. The loader reads them as strings, in first-seen order, so factor level
codes are stable across machines. Nothing is imputed, filtered or reordered: the
564 rows whose yield is exactly 0 are kept, because each one cost a real
experiment.
