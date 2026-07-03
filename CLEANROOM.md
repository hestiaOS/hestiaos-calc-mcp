# Clean Room Declaration — calc-mcp

## Purpose

This document declares the clean-room status of public projections of the
calc-mcp repository. Workflow documents, audit reports, and implementation
plans are explicitly excluded from public archives.

## Public export rules

The following categories are **excluded** from `git archive` public projections:

- **Workflow documents:** `IMPLEMENTATION-PLAN.md`, `audits/`
- **Secrets / credentials:** No `.env` files, API keys, tokens, or passwords.
- **Local configuration:** No editor configs, IDE files, or local overrides.
- **Private infrastructure:** No hostnames, private IPs, or internal repository URLs.
- **Runtime caches:** No `__pycache__`, `.pytest_cache`, `node_modules`, `.next`,
  `dist`, or `coverage` directories.
- **Build artifacts:** Native kernel outputs (`libcalc.dylib`, `*.o`, `build/`) are
  gitignored and excluded from public archives.
- **Private Git history:** Public projections start with a fresh Git repository
  initialised from the clean export. No remote is configured.

## What is included

- All source code (`calc_mcp/`, `tests/`)
- Build documentation (`idris/BUILD.md`)
- Public README, LICENSE (Apache 2.0), NOTICE
- Clean-room declaration (this file)
- `.gitattributes` with explicit export-ignore rules

## Native build requirement

The Idris kernel (`libcalc.dylib` / `libcalc.so`) must be built locally following
the step-by-step instructions in `idris/BUILD.md`. Without it, three
kernel-specific tests (supervisor, crash resilience, stdout hygiene) are
expected to fail.

## Verification

Each public projection passes these local gates before release:

- **Boundary check:** No secrets, private paths, or hostnames.
- **License audit:** LICENSE file present and Apache-2.0.
- **Clean-room documentation:** This file is present and accurate.
- **README quality:** Public-safe, functional, and free of internal references.
- **Full test suite:** Python test suite runs against the projection.

## Integrity

This build is intended for public release. If you suspect any sensitive data
or internal infrastructure information has been included, please raise an issue
before publishing.

No remote repository has been configured, no push has been executed, and no
online publication has occurred through this build process.
