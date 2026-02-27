# RIEnet PyPI Packaging Report

## Causa radice
1. Configurazione packaging duplicata tra `pyproject.toml` e `setup.py`, con metadata runtime (`install_requires`/`extras_require`) sovrascritti e warning setuptools durante build.
2. Workflow GitHub Actions pubblicava subito dopo il build senza `twine check`, quindi mancava un gate esplicito di validazione distribuzioni.
3. Primo anello che si rompe in locale (prima del packaging): `python -m venv` falliva per assenza di `ensurepip` nell'ambiente host.

Errore iniziale catturato:
```text
The virtual environment was not created successfully because ensurepip is not available.
```

## File modificati
- `.github/workflows/publish.yml`
- `pyproject.toml`
- `MANIFEST.in`
- `setup.py` (rimosso)
- `PYPI_RELEASE_REPORT.md` (questo report)

## Verifiche eseguite
Comandi eseguiti in venv pulita (`.venv-pypi`):

```bash
.venv/bin/python -m virtualenv --clear .venv-pypi
.venv-pypi/bin/python -m pip install -U pip build twine
.venv-pypi/bin/python -m build
.venv-pypi/bin/python -m twine check dist/*
.venv-pypi/bin/python -m compileall -f rienet
.venv-pypi/bin/python -m pip install dist/*.whl
.venv-pypi/bin/python -c "import rienet; print(rienet.__version__)"
```

Risultato:
- `python -m build` OK (sdist + wheel creati)
- `twine check dist/*` OK
- `compileall` OK
- install wheel + `import rienet` OK (`1.0.0`)

## Istruzioni per pubblicare
1. Push su `main` (o trigger manuale `workflow_dispatch`) del workflow `Build and publish Python package to PyPI`.
2. Il job esegue in ordine:
   - `python -m build`
   - `python -m twine check dist/*`
   - `pypa/gh-action-pypi-publish@release/v1` con Trusted Publishing (OIDC, `id-token: write`).
3. Assicurarsi che il progetto PyPI sia configurato per Trusted Publisher verso repo/workflow corretti.
