# Contributing to HDDFlyzer

Thank you for your interest in contributing to HDDFlyzer!
This project is maintained by the [NanoBiostructures Research Group](https://nanobiostructuresrg.github.io) at Tecnológico de Monterrey.

## How to contribute

### Reporting bugs
Open an issue on [GitHub Issues](https://github.com/NanoBiostructuresRG/hddflyzer/issues) with:
- A clear description of the problem
- A minimal reproducible example
- Your environment details (OS, Python version, RDKit version)

### Suggesting features
Open an issue with the `enhancement` label describing:
- The use case
- Why it would be useful beyond your specific workflow

### Submitting a pull request
1. Fork the repository
2. Create a branch from `dev/v0.1.0` during this readiness cycle:
   `git switch -c dev/your-feature`
3. Make your changes
4. Run the test suite: `pytest tests/`
5. Push your branch and open a pull request against `dev/v0.1.0`

Pull requests should pass CI before merge. User-facing changes should include
appropriate documentation and changelog updates.

## Development setup

```bash
git clone https://github.com/NanoBiostructuresRG/hddflyzer.git
cd hddflyzer
conda env create -f environment.yml
conda activate hddflyzer_env
pip install -e ".[dev]"
```

## Code style
This project follows [PEP 8](https://peps.python.org/pep-0008/). Please run `ruff` or `flake8` before submitting.

## Documentation and changelog
Update the README, documentation, examples, or API reference when behavior,
interfaces, commands, or examples change. Add an entry to `CHANGELOG.md` for
user-facing changes.

## Scientific and cheminformatics changes
For changes that affect SMILES standardization or cheminformatics assumptions,
explain or cite the relevant assumptions when appropriate. Add or update tests
and examples for changed standardization behavior, document RDKit-dependent
behavior when relevant, and preserve reproducibility of the workflow.

## Questions
Open an issue or contact the maintainer via the repository.
