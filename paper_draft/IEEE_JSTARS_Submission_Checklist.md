# IEEE JSTARS Submission Checklist

## 1. Metadata

- Confirm final title, author names, affiliations, and corresponding author email.
- Fill in the manuscript footer placeholders for received/revised dates if required by the template.
- Replace `[Funding Agency]` in `paper_draft/IEEE_JSTARS_Paper_Draft.md` with the real grant or project information.

## 2. Figures and Tables

- Verify that Figure 1 uses `results/runs/baseline_20260228_124857_seed42/run_miou_trajectory_compare.png` and remains readable after insertion into the IEEE template.
- Verify that Figure 2 uses `results/runs/baseline_20260228_124857_seed42/run_lambda_trajectory_compare.png` and that the legend is legible at journal column width.
- If needed, regenerate both figures with fewer curves or enlarged fonts for publication quality.
- Convert Markdown image references into formal IEEE figure environments when moving to LaTeX or Word.
- Check that Table I, Table II, and Table III numbering is consistent after template conversion.

## 3. Language and Style

- Run a final grammar pass with Grammarly, LanguageTool, or a professional academic editing service.
- Standardize terminology: use either `LLM agent` or `LLM-Agent` consistently throughout the final version.
- Standardize metric style: `mIoU`, `F1-score`, and `ALC` should be formatted consistently in text, tables, and captions.
- Reduce overclaiming: keep the current phrasing that AAL-SD has the best final accuracy and second-best ALC in the seed-42 run.

## 4. Experimental Integrity

- Double-check all reported seed-42 values against `results/runs/baseline_20260228_124857_seed42/` before submission.
- If possible, add multi-seed mean and standard deviation in a revision or supplementary material.
- Clarify in the methods or discussion whether seed 42 is the primary reporting seed or part of a broader replication setting.
- Consider adding one paragraph on statistical robustness if additional seed runs are available.

## 5. References

- Reformat all references to the exact IEEE JSTARS style in the final submission template.
- Add missing page numbers, issue numbers, DOIs, and publisher details where available.
- Verify that every in-text citation appears in the bibliography and vice versa.
- Replace arXiv-only citations with published versions if any have since appeared in journals or conferences.

## 6. Recommended Next Improvements

- Add a short supplementary note or appendix describing the prompt template and tool interface for the LLM controller.
- Add a compact qualitative figure showing selected landslide patches or query examples if available.
- Add a short paragraph explaining the deployment trade-off between higher final accuracy and slightly lower ALC than Wang-style.
- Prepare a code/data availability statement for the final manuscript.

## 7. Files to Use

- Main draft: `paper_draft/IEEE_JSTARS_Paper_Draft.md`
- Outline: `paper_draft/IEEE_JSTARS_Outline.md`
- Figure 1 source: `results/runs/baseline_20260228_124857_seed42/run_miou_trajectory_compare.png`
- Figure 2 source: `results/runs/baseline_20260228_124857_seed42/run_lambda_trajectory_compare.png`
