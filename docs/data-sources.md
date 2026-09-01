# Raw data sources and local layout

The code performs no automatic downloads. Download each corpus from its
official source, preserve the original files, and place them below `ori_data/`.
Notebook 01 performs a fast count/size audit on every run; set `FULL_RAW_AUDIT`
to `True` once to calculate a content-manifest SHA-256 for each source.

```text
ori_data/
├── Enron/maildir/<person>/<mailbox>/<message files>
├── Nazario/<mbox files>
├── PhishingPot/PhishingPot/*.eml
└── SpamAssassin/
    ├── easy_ham/<message files>
    ├── easy_ham_2/<message files>
    └── hard_ham/<message files>
```

## Source and label mapping

| Local source | Official source | Original material used | Experimental label |
|---|---|---|---|
| `Enron` | Enron Email Dataset | Messages below `maildir` | legitimate (0) |
| `SpamAssassin` | SpamAssassin public mail corpus | `easy_ham`, `easy_ham_2`, `hard_ham` only | legitimate (0) |
| `Nazario` | Nazario Phishing Corpus | mbox messages; licence/readme excluded | phishing (1) |
| `PhishingPot` | Phishing Pot | `.eml` samples | phishing (1) |

The SpamAssassin **spam** folders are deliberately not used. This prevents a
spam label from being silently treated as phishing. The experimental labels
are source-derived rather than manually re-annotated, and this limitation must
be stated in the report.

## Verification

Notebook 01 records paths, file counts and byte counts in
`artifacts/data/raw-audit.json`. A full audit additionally hashes each relative
file path and file SHA-256 into one source-level digest. The processing audit
then records exclusions, exact duplicates, conflicts, source balance, split
sizes and representation row counts. A successful default rebuild is expected
to produce 3,195 samples from each source: 12,780 in total, split into 8,948
train, 1,916 validation and 1,916 test records. If the local corpus release
differs, stop and explain the discrepancy instead of forcing these totals.

## References

Cohen, W.W. (2015) *Enron Email Dataset*. Carnegie Mellon University, May 2015
release. Available at: <https://www.cs.cmu.edu/~enron/> (Accessed: 4 August 2026).

Nazario, J. (n.d.) *Nazario Phishing Corpus*. Available at:
<https://monkey.org/~jose/phishing/> (Accessed: 4 August 2026).

Peixoto, R.F. and contributors (n.d.) *Phishing Pot: A collection of phishing
samples for researchers and detection developers*. GitHub repository.
Available at: <https://github.com/rf-peixoto/phishing_pot> (Accessed: 4 August 2026).

Apache SpamAssassin Project (2006) *SpamAssassin public mail corpus*. Available
at: <https://spamassassin.apache.org/old/publiccorpus/readme.html> (Accessed: 4 August 2026).
