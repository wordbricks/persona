# Releasing `@wordbricks/persona`

Releases are published from GitHub Actions with npm trusted publishing (OIDC).
The workflow does not use a long-lived `NPM_TOKEN`.

## One-time setup

1. In the GitHub repository, create an environment named `npm`.
   Add required reviewers if releases should require manual approval.
2. In the npm package settings for `@wordbricks/persona`, add a GitHub Actions
   trusted publisher with these exact values:
   - Organization: `wordbricks`
   - Repository: `persona`
   - Workflow filename: `publish.yml`
   - Environment: `npm`
   - Allowed action: `npm publish`
3. After the first OIDC release succeeds, configure npm publishing access to
   require 2FA and disallow tokens, then revoke obsolete automation tokens.

## Release a version

1. Update `package.json` and `CHANGELOG.md` in a pull request and merge it.
2. Confirm CI passes on `main`.
3. Create and push the matching version tag from the updated `main` commit:

   ```sh
   git switch main
   git pull --ff-only origin main
   git tag -a v0.2.1 -m "Release v0.2.1"
   git push origin v0.2.1
   ```

The publish workflow rejects a tag that does not exactly match the
`package.json` version. It installs with the frozen lockfile, then runs the
typecheck, tests, build, package dry-run, and `npm publish --access public`.

Do not reuse or move a release tag. Bump the package version and create a new
tag if a release needs a follow-up fix.
