# Execution Plan: Project Page — Fill In Citation BibTeX

## Metadata

- **Created**: 2026-06-15
- **Status**: Active (Blocked — waiting on NeurIPS 2026 acceptance / publication)
- **Priority**: Medium
- **Type**: Docs / Project Page

## Objective

Replace the placeholder citation block on the project page with the final
BibTeX entry once the venue and bibliographic details are confirmed.

## Context

- The paper has **not** been published at NeurIPS (or any venue) yet.
- Filling in a fabricated `@inproceedings{zhu2026steering, ... NeurIPS 2026}`
  entry would misrepresent the publication status.
- `docs/index.html` currently renders an empty placeholder so the section
  exists without a fake citation. This plan tracks the follow-up.

## Current State (in `docs/index.html`)

```html
<pre><code id="bibtex-code">% Citation will be added once the paper is published.</code></pre>
```

## Steps (when unblocked)

1. Confirm final publication metadata:
   - Accepted venue (NeurIPS 2026 or other) and proceedings details.
   - Final title, author order, affiliations.
   - Assigned DOI / arXiv ID (if applicable).
2. Author the canonical BibTeX entry, e.g.:

   ```bibtex
   @inproceedings{zhu2026steering,
     title     = {Not All Tokens Are Equally Useful for Steering: Robust Directions and Prefix Steering},
     author    = {Zhu, Xudong and Zhu, Zhihui},
     booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
     year      = {2026}
   }
   ```

   (Adjust key/fields to match the actual venue and DOI.)

3. Replace the placeholder `<code id="bibtex-code">...</code>` contents in
   `docs/index.html` with the real entry. Keep the `id="bibtex-code"` so the
   existing copy-to-clipboard JS in `static/js/main.js` keeps working.
4. Mirror the same entry into `README.md`'s `## Citation` block.
5. Sanity-check the "Copy" button copies the new entry correctly.

## Acceptance Criteria

- [ ] BibTeX entry reflects the confirmed venue and metadata (no placeholder
      fields, no fabricated acceptance).
- [ ] Copy-to-clipboard still works on the project page.
- [ ] `README.md` citation block matches `docs/index.html`.
