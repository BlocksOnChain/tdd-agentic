# Agent notes (this workspace)

## Where documentation lives

- **Do not** point agents at `node_modules/` for guides or API references. Vendored packages are not a stable documentation surface and should stay out of prompts, skills, and RAG context.
- Put **project-specific** and **framework** notes under **`docs/`** in this workspace (for example `docs/nextjs.md`, `docs/conventions.md`, `docs/api.md`). The Researcher should author or refresh these and persist them into RAG with `rag_ingest_text`.
- For **canonical** vendor documentation, use **stable HTTPS URLs** (for example [Next.js docs](https://nextjs.org/docs)) and copy only the behavior that matters into `docs/` for offline, version-aligned reference.

## Framework stacks (e.g. Next.js)

Installed behavior is defined by this repo’s `package.json` and lockfile. If stack knowledge may be outdated, prefer `docs/<framework>.md` plus official sites — never paths inside `node_modules/`.
