# Froggy Team Website

Source for the [Froggy team website](https://microsoft.github.io/debug-gym/),
built with Jekyll and hosted on GitHub Pages.

This branch contains only the website and its maintenance tooling. The debugging
environment and agent code are maintained on the
[`main` branch](https://github.com/microsoft/debug-gym/tree/main).

## Repository Layout

```text
docs/                   Jekyll source, blog posts, report data, and assets
tests/docs/             Website rendering tests
.github/workflows/      Website build, test, and formatting checks
requirements-dev.txt    Python tools for website development only
```

The license, code of conduct, security policy, and support information remain at
the repository root.

## Setup and Build

Use the Ruby version specified in `docs/.ruby-version` and install Bundler.
Native gems also require your platform's compiler and build tools.

From the repository root:

```sh
cd docs
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll build
```

The generated site is written to `docs/_site/`. Building the website does not
require the Python application, model dependencies, or API credentials.

See the [Jekyll guide](docs/README-JEKYLL.md) for local previews, blog authoring,
technical report cards, and asset conventions.

## Development Checks

Python 3.12 is used only for website tests and formatting. From the repository
root, create a virtual environment and install the small development tool set:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pre_commit run --all-files
```

The rendering tests also require the site gems installed above and `bundle`
available on `PATH`. To enable the formatting hooks for local commits, run
`python -m pre_commit install`.

## Publishing

Open website pull requests against `gh-page`. GitHub Pages is configured to build
the `/docs` directory on that branch; feature-branch pushes do not deploy the
website. Merging into `gh-page` triggers the Pages deployment.

Keep the `docs/` source directory and `/debug-gym` base URL unchanged so existing
page and asset URLs continue to work.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
