# MSR Montréal Froggy Team - Jekyll Site

This site is built with Jekyll, allowing you to write content in Markdown while maintaining the same beautiful design.

## Setup

### Prerequisites
- Ruby as specified in `.ruby-version`
- Compiler and build tools for native gem extensions
- Bundler (`gem install bundler`)

### Installation

1. Install dependencies (installs gems locally to avoid sudo prompts on macOS):
```bash
cd docs
bundle config set --local path vendor/bundle
bundle install
```

2. Start the Jekyll development server:
```bash
bundle exec jekyll serve --port 4000
```

3. Visit `http://localhost:4000/debug-gym/` in your browser

## Site Structure

```
docs/
+-- .ruby-version        # Ruby version for development and CI
+-- Gemfile              # Jekyll dependencies
+-- Gemfile.lock         # Locked gem versions
+-- _config.yml          # Site and collection configuration
+-- _data/               # News and standalone report cards
+-- _includes/           # Navigation and reusable components
+-- _layouts/            # Default, blog, and project templates
+-- _posts/              # Markdown blog posts
+-- figures/             # Blog figures and interactive example data
+-- static/              # Stylesheets, scripts, images, and pending papers
+-- index.html           # Auto-generated homepage feed
+-- README-JEKYLL.md      # Authoring and development guide
```

The optional `_projects/` collection is available for new project pages.

## Writing Content

### Adding a Standalone Technical Report

Reports can have their own homepage cards, separate from News and Blog Posts.
Each card shows a Technical Report badge, the date, the title, an optional summary,
and a direct paper link. No blog or project page is required.

1. Use the report's direct arXiv PDF URL if available. Otherwise, copy the PDF into
   `docs/static/papers/`.
2. Add an entry to `docs/_data/papers.yml` with a title, date, and link, plus an
   optional description:

```yaml
- title: "Report title"
  date: 2026-09-07
  description: "A short summary of the report."
  link: "https://arxiv.org/pdf/your-paper-id"
```

For a hosted PDF, use `link: "/static/papers/your-report.pdf"`. The template applies
`relative_url` so the link works locally and under the GitHub Pages `baseurl`.
Full external URLs are also supported and are left unchanged.

The entry appears in the homepage feed in reverse date order, after any entries
marked `always_top: true`. Set `draft: true` to hide a report.

Once the report is available on arXiv, update the entry's `link` to its PDF URL.
Remove the redundant hosted PDF after updating all site references to it.

### Creating a New Project Page

1. Create a new file in `_projects/` (e.g., `_projects/my-project.md`)

2. Add front matter and content:

```markdown
---
layout: project
title: "My Project"
title_html: '<code>my-project</code>'  # Optional: for code formatting
description: "A short description"
authors: 'Author Name'
email: "contact@example.com"
affiliation: "Microsoft Research"
github_url: "https://github.com/..."
arxiv_url: "https://arxiv.org/pdf/..."
team_logo: "/static/images/my-team-logo.png"  # Optional: shown below authors
bibtex: |
  @article{...}
---

## Overview

Your content here in **Markdown**!

### Subheading

- Bullet points
- More content

## Another Section

More markdown content...
```

3. The project will automatically appear on the index page!

### Creating a New Blog Post

1. Create a new file in `_posts/` with the format: `YYYY-MM-DD-title.md`

2. Add front matter and content:

```markdown
---
layout: blog-post
title: "Your Blog Post Title"
date: 2025-01-15
author: "Author Name"
reading_time: 8
tags: ["AI", "Debugging", "Research"]
description: "A brief description"
paper_url: "https://arxiv.org/pdf/your-paper"
# paper_local: "/static/papers/your-paper.pdf"
authors:
  - name: "First Author"
    role: "Researcher"
  - name: "Second Author"
    role: "Engineer"
---

Your blog post content in **Markdown**!

## Section Header

Content here...

```python
# Code blocks work great!
def hello():
    print("Hello, world!")
```

### Subsection

More content...
```

3. The blog post will automatically appear on the index page!

Posts with `published: false` are excluded from production builds. Use
`--unpublished` only for local review, then remove that flag or set it to `true`
before merging a post for publication.

**Paper links:**

Use `https://arxiv.org/pdf/<paper-id>` for all arXiv links, including paper buttons
and inline citations, so they open the PDF directly.

- Use `arxiv_url` for arXiv links or `paper_url` for other external paper links.
- To host a PDF locally, drop it in `docs/static/papers/` and reference it with `paper_local: "/static/papers/<file>.pdf"`.
- Once an arXiv version is available, remove `paper_local` and the redundant PDF after updating any other references.
- Set `paper_url: "#"` to show a disabled "Paper link coming soon" button; omit all paper-link fields to hide it.

### Team data (optional)

The legacy Team page has been retired, and `_data/team.yml` has been removed. If you’d like blog posts (or other templates) to look up author metadata, you can recreate the file with entries like:

```yaml
- name: "Ada Lovelace"
  role: "Research Scientist"
  affiliation: "Microsoft Research Montréal"
  links:
    - label: "Scholar"
      url: "https://scholar.google.com/..."
```

Layouts are resilient if the file is missing, so only add it back when you need the extra data.

## Markdown Features

All standard Markdown is supported:

- **Bold text**
- *Italic text*
- `Code inline`
- [Links](https://example.com)
- Images: `![Alt text]({{ '/static/images/example.png' | relative_url }})`
- Lists (ordered and unordered)
- Headers (h2, h3, etc.)
- Code blocks with syntax highlighting
- Blockquotes
- Tables

### Referencing Static Assets

When embedding images or gifs, always use the `relative_url` filter so links work locally and on GitHub Pages:

```markdown
![Overview diagram]({{ '/static/images/overview.png' | relative_url }})

<img src="{{ '/static/images/demo.gif' | relative_url }}" alt="Demo" />
```

All files should live in `static/images/`.

## Customizing Design

- **CSS**: Edit `static/css/custom.css` for global styles
- **Layouts**: Modify files in `_layouts/` to change page structure
- **Navigation**: Edit `_includes/nav.html` to update the navbar
- **Colors/Branding**: Update CSS variables in `custom.css`

## Building for Production

For GitHub Pages:

1. Open a pull request against `gh-page`.
2. After it is merged, GitHub Pages builds and deploys the branch's `/docs` folder.

Feature-branch pushes do not publish the website. Keep this source directory and
the `/debug-gym` base URL unchanged.

For manual build:

```bash
bundle exec jekyll build
```

Output will be in `_site/` directory.

### Homepage Rendering Tests

After installing the site bundle, use a Python virtual environment and run these
commands from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pre_commit run --all-files
```

They render the real Jekyll homepage and blog pages to cover report cards, direct
arXiv PDF links, hosted paper links, feed ordering, and draft visibility.
Python is needed only for these development checks, not for building or deploying
the website.

## Key Features

✅ **Write in Markdown** - No HTML required for content  
✅ **Auto-generated index** - Projects and blog posts automatically listed  
✅ **Preserved design** - Same beautiful UI as before  
✅ **BibTeX support** - Copy-to-clipboard functionality  
✅ **Responsive** - Mobile-friendly design  
✅ **SEO-friendly** - Meta tags and structured data  

## Troubleshooting

### Jekyll not found
```bash
bundle config set --local path vendor/bundle
bundle install
```

### Port already in use

Stop your own preview with Ctrl+C, or use a different port:

```bash
bundle exec jekyll serve --port 4001
```

### Videos stop early in a local preview

Some WEBrick versions mishandle conditional byte-range requests, causing videos
to stop even when the files are intact. In that case, build with
`bundle exec jekyll build --watch` and serve the generated files with a
range-capable HTTP server mounted at `/debug-gym/`. GitHub Pages does not use the
local WEBrick server.

### Changes not showing
- Hard refresh browser (Cmd+Shift+R on Mac)
- Restart Jekyll server
- Check `_site/` is being regenerated
- Remove cached build artifacts if needed: delete `docs/_site/`, `docs/.jekyll-cache/`, and `docs/vendor/` before re-running the build

## File Organization Tips

### Projects (_projects/)
- Use descriptive filenames: `debug-gym.md`, `bugpilot.md`
- Set `status: "coming-soon"` for unreleased projects
- Include `github_url`, `arxiv_url`, `paper_url` as needed

### Blog Posts (_posts/)
- Follow naming: `YYYY-MM-DD-title.md`
- Use descriptive titles and tags
- Estimate `reading_time` in minutes
- Add multiple authors if needed

### Images
- Store in `static/images/`
- Reference them with `{{ '/static/images/...' | relative_url }}` so the site `baseurl` is applied automatically
- Use descriptive alt text for accessibility

## Questions?

Contact: debug-gym@microsoft.com
