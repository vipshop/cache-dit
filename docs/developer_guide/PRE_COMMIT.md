# Prepare before commit

## 👨‍💻 Run Pre-commit

Before submitting code, configure pre-commit, for example:

```bash
# fork vipshop/cache-dit to your own github page, then:
git clone git@github.com:your-github-page/your-fork-cache-dit.git
cd your-fork-cache-dit && git checkout -b dev
# update submodule
git submodule update --init --recursive --force
# install pre-commit
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

## 👨‍💻 Add a new feature

```bash
# feat: support xxx-cache method
# add your commits
git add .
git commit -m "support xxx-cache method"
git push
# then, open a PR from your personal branch to cache-dit:main
```

## 👨‍💻 Check MKDocs

Please also check the mkdocs build status on your local branch.
```bash
pip3 install -e ".[docs]"
mkdocs build --strict
mkdir serve # Then check the docs
```

Ensure that your new commits do not break the mkdocs build process.

```bash
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: /workspace/dev/vipshop/cache-dit/site
INFO    -  Documentation built in 0.97 seconds
```
