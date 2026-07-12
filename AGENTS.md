# Repository Instructions

## npm releases

`@wordbricks/persona` is published by `.github/workflows/publish.yml` through
npm trusted publishing (OIDC). The npm trusted publisher and the GitHub
environment named `npm` are already configured. `RELEASING.md` is the human
release runbook.

Only start a release when the user explicitly asks to publish or release a
version. A request to implement, merge, or bump a version by itself does not
authorize creating a release tag or publishing to npm.

When a release is authorized:

1. Update the version in `package.json` and add the same version and release
   date to `CHANGELOG.md`. Do not change `bun.lock` solely for a package version
   bump because this workspace lockfile does not store the root version.
2. Run all release checks locally:

   ```sh
   bun install --frozen-lockfile
   bun run typecheck
   bun run test
   bun run build
   npm pack --dry-run
   ```

3. Commit and merge the versioned release changes before tagging. Confirm the
   target commit is on `origin/main`, the worktree is clean, and CI on `main`
   succeeded.
4. Confirm that the version is not already present on npm. Never overwrite an
   existing npm version.
5. Create one annotated tag whose name exactly matches `v` plus the
   `package.json` version, and push only that tag. For example:

   ```sh
   git switch main
   git pull --ff-only origin main
   git tag -a v0.2.2 -m "Release v0.2.2"
   git push origin v0.2.2
   ```

6. Monitor the `Publish npm package` GitHub Actions run through completion.
   Verify both the exact published version and the `latest` dist-tag in the npm
   registry before reporting success.

Never run `npm publish` locally, add an `NPM_TOKEN`, or bypass the OIDC publish
workflow. Never reuse, move, or force-push a release tag. If a release fails
after the version has reached npm, prepare a new patch version instead.
