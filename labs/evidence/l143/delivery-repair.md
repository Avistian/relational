# Local delivery repair

The first full browser/delivery check passed the interactive, viewport, print, no-JS, copied-site and gallery stages, then failed its deterministic-build assertion. Comparing the generated notebook against the staged executed artifact showed discarded cell execution timestamps and notebook metadata. The builder was corrected to retain metadata for unchanged code; the executed artifact was restored from the task's staged copy. No scientific code, predictions or cloud execution changed. Final actual outcome is in labs/_delivery_l143_results.json.
