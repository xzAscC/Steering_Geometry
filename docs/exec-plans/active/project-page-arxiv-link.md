# Execution Plan: Project Page — Add arXiv Link

## Metadata

- **Created**: 2026-06-15
- **Status**: Active (Blocked — waiting on arXiv submission)
- **Priority**: Medium
- **Type**: Docs / Project Page

## Objective

Replace the placeholder (disabled) arXiv button on the project page with a real
link to the arXiv preprint once it is posted.

## Context

- The paper is not yet on arXiv.
- `docs/index.html` currently renders a disabled "Coming soon" arXiv button so
  the layout stays stable in the meantime.
- This plan tracks the follow-up; it is **not** actionable until the arXiv
  abstract URL exists.

## Current State (in `docs/index.html`)

```html
<a class="button is-link is-rounded is-static" title="Coming soon" aria-disabled="true">
  <span class="icon"><i class="ai ai-arxiv" aria-hidden="true"></i></span>
  <span>arXiv</span>
</a>
```

## Steps (when unblocked)

1. Obtain the arXiv abstract URL (e.g. `https://arxiv.org/abs/2026.XXXXX`).
2. In `docs/index.html`, replace the static button with an active link:

   ```html
   <a class="button is-link is-rounded" href="https://arxiv.org/abs/2026.XXXXX"
      target="_blank" rel="noopener">
     <span class="icon"><i class="ai ai-arxiv" aria-hidden="true"></i></span>
     <span>arXiv</span>
   </a>
   ```

3. Optionally update the README `arXiv preprint` sentence with the same URL.
4. Verify the page locally and rebuild/deploy the GitHub Pages site.

## Acceptance Criteria

- [ ] arXiv button links to the live preprint (opens in a new tab).
- [ ] Button is no longer marked `is-static` / `aria-disabled`.
- [ ] No other layout regressions in the links row.
