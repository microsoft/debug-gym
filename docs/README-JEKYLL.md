# MSR Montréal Froggy Team - Jekyll Site

This site is built with Jekyll, allowing you to write content in Markdown while maintaining the same beautiful design.

## Setup

### Prerequisites
- Ruby (2.6 or higher recommended)
  - conda install conda-forge::ruby
  - conda install -c conda-forge gcc_linux-64 gxx_linux-64 make openssl -y
- Bundler (`gem install bundler`)

### Installation

1. Install dependencies (installs gems locally to avoid sudo prompts on macOS):
```bash
cd docs
bundle install --path vendor/bundle
```

2. Start the Jekyll development server:
```bash
bundle exec jekyll serve --port 4000
```

3. Visit `http://localhost:4000/debug-gym/` in your browser

## Site Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── _layouts/             # Page templates
│   ├── default.html      # Base layout with nav
│   ├── project.html      # Research project pages
│   └── blog-post.html    # Blog post pages
├── _data/                # Structured data used across pages (add your own files as needed)
├── _includes/            # Reusable components
│   └── nav.html          # Navigation bar
├── _projects/            # Research project pages (Markdown)
│   ├── debug-gym.md
│   ├── bugpilot.md
│   └── gistify.md
├── _posts/               # Blog posts (Markdown)
│   └── 2025-01-15-building-ai-debugging-agents.md
│   └── 2025-10-22-bug-pilot.md
│   └── 2025-10-28-gistify.md
├── index.html            # Landing page (auto-generates from collections)
├── static/               # CSS, JS, pdf
└── figures/              # figures for post
    ├── bug-pilot         # folder for figures
    └── gistify           # folder for figures



```

## Writing Content

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
arxiv_url: "https://arxiv.org/abs/..."
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
paper_url: "https://arxiv.org/abs/your-paper"
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

**Paper links:**

- Use `paper_url` for any external link (arXiv, conference page, etc.).
- To host a PDF locally, drop it in `docs/static/papers/` and reference it with `paper_local: "/static/papers/<file>.pdf"`.
- Leave both fields blank (or set `paper_url: "#"`) to surface a disabled “Paper link coming soon” button.

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

1. Push your changes to the repository
2. GitHub Pages will automatically build and deploy

For manual build:

```bash
bundle exec jekyll build
```

Output will be in `_site/` directory.

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
bundle install --path vendor/bundle
```

### Port already in use
```bash
# Kill existing server
lsof -ti:4000 | xargs kill

# Or use a different port
bundle exec jekyll serve --port 4001
```

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
