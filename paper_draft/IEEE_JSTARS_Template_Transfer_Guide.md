# IEEE JSTARS Template Transfer Guide

## 1. Recommended Transfer Order

1. Copy title, authors, affiliations, corresponding author, and funding block from `paper_draft/IEEE_JSTARS_Paper_Draft.md` into the official IEEE template.
2. Copy the abstract and `Index Terms` section.
3. Copy the main body from Section I to Section VII.
4. Insert Table I, Table II, and Table III using the journal template's table style.
5. Insert Figure 1 and Figure 2 using the image files listed below.
6. Append Acknowledgment, Data Availability Statement, and Code Availability Statement if the target submission workflow allows them in the manuscript file.
7. Reformat references into exact IEEE style using EndNote, Zotero, Mendeley, or manual cleanup.

## 2. Figure Assets

- Figure 1: `results/runs/baseline_20260228_124857_seed42/run_miou_trajectory_compare.png`
- Figure 2: `results/runs/baseline_20260228_124857_seed42/run_lambda_trajectory_compare.png`

## 3. Table Assets

- Table I: overall performance comparison
- Table II: ablation study results
- Table III: sample agent decision explanations

## 4. Required Manual Fill-ins

- Author names
- Affiliations
- Corresponding author email
- Funding information
- ORCID identifiers if required
- Public code repository URL

## 5. Recommended Final Polishing

- Shorten some paragraphs in Sections I and V after transfer to meet page limits.
- If the figure legends appear crowded, redraw the plots with fewer curves or highlight only the main compared methods.
- If multi-seed statistics are available later, add a short robustness paragraph or supplementary table.
