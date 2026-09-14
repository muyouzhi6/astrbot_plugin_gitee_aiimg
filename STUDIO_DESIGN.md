# Studio delivery contract

- Embed `pages/studio` in AstrBot Pages and use authenticated plugin APIs.
- Keep existing configuration as the single provider/routing source. Mask credentials, preserve unchanged secrets, reject stale writes, back up before saving.
- Fetch model IDs from the selected provider. Preserve manual IDs. Do not infer parameter capabilities from model listing.
- Render draw/edit/selfie/video chains as sortable rows. Support pointer, touch and explicit move buttons. Preserve provider IDs and per-link overrides.
- Store private image assets and prompt history outside the plugin source. Keep originals and thumbnails, deduplicate content, paginate history, never expose arbitrary paths.
- Maintain named characters with multiple appearances and explicit sender/session access. Resolve `me` only from configured sender identity, and `self` only from the selected bot appearance.
- Bind each subject to numbered reference images and its own wardrobe. Apply cached life-scheduler outfit only to the bot subject unless the current request explicitly overrides it.
- Require all named subjects and ordered reference support. Reject missing, ambiguous, unauthorized and duplicate identity references before starting generation.
- Snapshot identity references and schedule before queue acceptance. Do not re-resolve mutable characters during background execution.
- Archive successful results from existing draw/edit routers, including prompt, provider and model. Never claim unavailable historical prompts were recovered.
- Persist canvas layers, transforms, ordering and prompt. Support upload, gallery reuse, pan/zoom, touch, export, undo/redo and generation from selected assets.
- Reuse the existing provider scheduler and durable task manager. Do not resubmit interrupted generation automatically. Separate local cancellation from upstream billing.
- Verify API validation, stale writes, identity isolation, clothing isolation, model fetch, job deduplication, restart behavior and desktop/mobile user paths before release.
