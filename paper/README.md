## Plot Data

The data used to create the figures of the arXiv preprint article can be
found in the `results.tar.bz2.part*` archive. Unpack the archive and run the
file `plots.py`.

```bash
cat results.tar.bz2.part_* | tar -xjf -
```

To compress back the data, run:

```bash
tar -cjf - results | split -b 50M - results.tar.bz2.part_
```
