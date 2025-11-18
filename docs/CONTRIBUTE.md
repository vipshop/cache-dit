# Developer Guide

## 👨‍💻Pre-commit

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

## 👨‍💻Add a new feature

```bash
# feat: support xxx-cache method
# add your commits
git add .
git commit -m "support xxx-cache method"
git push
# then, open a PR from your personal branch to cache-dit:main
```
