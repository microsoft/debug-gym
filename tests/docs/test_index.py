import json
import re
import subprocess
from html import unescape
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
REPORT_TITLE = "FrogNano: Training a 4B Coding Agent via Online Task Synthesis"
REPORT_URL = "https://arxiv.org/pdf/2609.07925"
LOCAL_REPORT_PATH = "/static/papers/example-report.pdf"
REPORT_DESCRIPTION = (
    "We introduce FrogNano, a 4B coding agent post-trained exclusively with "
    "reinforcement learning on synthetic software engineering tasks. Adapting "
    "tasks to the agent's evolving capabilities enables competitive performance "
    "without distillation from larger models."
)


def render_page(page_url="/", **options):
    result = subprocess.run(
        [
            "bundle",
            "exec",
            "ruby",
            "-rjekyll",
            "-rjson",
            "-e",
            """
options = JSON.parse(STDIN.read)
config = Jekyll.configuration(
  "source" => Dir.pwd, "quiet" => true, "disable_disk_cache" => true
)
config["baseurl"] = options["baseurl"] if options.key?("baseurl")
site = Jekyll::Site.new(config)
site.reset
site.read
site.data["papers"] = options["papers"] if options.key?("papers")
site.data["news"] = options["news"] if options.key?("news")
options.fetch("projects", []).each_with_index do |data, index|
  collection = site.collections.fetch("projects")
  document = Jekyll::Document.new(
    File.join(site.source, "_projects", "example-#{index}.md"),
    site: site, collection: collection
  )
  document.data["date"] = Time.utc(2025, 1, 1)
  document.data["slug"] = "example-#{index}"
  document.data.merge!(data)
  collection.docs << document
end
site.generate
site.render
pages = site.pages + site.posts.docs
puts pages.find { |page| page.url == options.fetch("page_url") }.output
""",
        ],
        cwd=DOCS,
        input=json.dumps({"page_url": page_url, **options}),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("baseurl", ["", "/debug-gym", "/preview"])
def test_frognano_card_links_to_paper_weights_and_harness(baseurl):
    homepage = render_page(baseurl=baseurl)
    cards = re.split(r'<div class="project-card"[^>]*>', homepage)[1:]
    report_cards = [card for card in cards if REPORT_TITLE in card]

    assert len(report_cards) == 1
    card = report_cards[0]
    assert "content-type-paper" in card
    assert "Technical Report" in card
    assert "content-type-news" not in card
    assert f"<p>{REPORT_DESCRIPTION}</p>" in unescape(card)
    assert (
        card.index("</h3>")
        < card.index("<p>")
        < card.index('<div class="project-links">')
    )
    assert card.count('class="button ') == 3
    assert f'href="{REPORT_URL}"' in card
    for label, url, style, icon in [
        (
            "FrogNano-4B",
            "https://huggingface.co/microsoft/FrogNano-4B-2609",
            "is-success",
            "fas fa-frog",
        ),
        (
            "Leaf Harness",
            "https://github.com/microsoft/FrogNano",
            "is-dark",
            "fab fa-github",
        ),
    ]:
        buttons = [
            anchor
            for anchor in re.findall(r"<a\b[^>]*>.*?</a>", card, re.S)
            if f"<span>{label}</span>" in anchor
        ]
        assert len(buttons) == 1
        assert f'href="{url}"' in buttons[0]
        assert f'class="button {style}"' in buttons[0]
        assert f'class="{icon}"' in buttons[0]
        assert 'target="_blank"' in buttons[0]
        assert 'rel="noopener noreferrer"' in buttons[0]
    assert not (DOCS / "static/papers/frognano_technical_report.pdf").exists()
    assert f'href="{baseurl}/blog/2026/08/negative-pi/"' in homepage
    assert "Apply Now" not in homepage


@pytest.mark.parametrize("baseurl", ["", "/debug-gym", "/preview"])
def test_report_card_still_supports_hosted_pdf(baseurl):
    homepage = render_page(
        baseurl=baseurl,
        papers=[
            {
                "title": "Local technical report",
                "date": "2026-09-14",
                "link": LOCAL_REPORT_PATH,
            }
        ],
    )

    assert f'href="{baseurl}{LOCAL_REPORT_PATH}"' in homepage
    card = next(
        card
        for card in re.split(r'<div class="project-card"[^>]*>', homepage)[1:]
        if "Local technical report" in card
    )
    assert card.count('class="button ') == 1
    assert "<span>FrogNano-4B</span>" not in card
    assert "<span>Leaf Harness</span>" not in card


@pytest.mark.parametrize(
    "arxiv_url,local_pdf",
    [
        ("https://arxiv.org/pdf/2510.19898", "BugPilot_arxiv.pdf"),
        ("https://arxiv.org/pdf/2510.26790", "Gistify_arxiv.pdf"),
    ],
)
def test_blog_cards_use_arxiv_instead_of_local_pdfs(arxiv_url, local_pdf):
    homepage = render_page()

    assert f'href="{arxiv_url}"' in homepage
    assert not (DOCS / "static/papers" / local_pdf).exists()


def test_report_links_and_feed_ordering():
    arxiv_url = "https://arxiv.org/pdf/example"
    homepage = render_page(
        papers=[
            {
                "title": "Older report",
                "date": "2020-01-01",
                "link": LOCAL_REPORT_PATH,
            },
            {
                "title": "Draft report",
                "date": "2022-01-01",
                "link": LOCAL_REPORT_PATH,
                "draft": True,
            },
            {
                "title": "Newer report",
                "date": "2021-01-01",
                "link": LOCAL_REPORT_PATH,
            },
            {
                "title": "Featured report",
                "date": "2019-01-01",
                "link": arxiv_url,
                "always_top": True,
            },
        ]
    )
    cards = re.split(r'<div class="project-card"[^>]*>', homepage)[1:]

    assert "Featured report" in cards[0]
    assert f'href="{arxiv_url}"' in cards[0]
    assert "<p>" not in cards[0]
    assert "Draft report" not in homepage
    assert homepage.index("Newer report") < homepage.index("Older report")
    assert "Read Blog Post" in homepage


def test_homepage_without_reports_keeps_existing_posts():
    homepage = render_page(papers=[])

    assert "content-type-paper" not in homepage
    assert 'href="/debug-gym/blog/2026/08/negative-pi/"' in homepage
    assert "No updates yet" not in homepage


@pytest.mark.parametrize("baseurl", ["", "/debug-gym", "/preview"])
def test_homepage_cards_have_unique_linked_anchors(baseurl):
    homepage = render_page(baseurl=baseurl)
    cards = re.findall(
        r'<div class="project-card" id="([^"]+)">(.*?)(?=<div class="project-card"|</section>)',
        homepage,
        re.S,
    )
    anchors = [anchor for anchor, _ in cards]

    assert len(cards) == homepage.count('<div class="project-card"')
    assert len(anchors) == len(set(anchors))
    assert "frognano" in anchors
    assert "post-blog-2025-10-bug-pilot" in anchors
    for anchor, card in cards:
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", anchor)
        assert f'class="project-card__permalink" href="#{anchor}"' in card
        assert card.index('class="project-card__permalink"') < card.index("</h3>")


def test_anchors_cover_news_reports_and_projects():
    homepage = render_page(
        news=[
            {"title": "Team News", "date": "2025-01-01"},
            {
                "title": "Custom news",
                "title_html": "<code>Custom news</code>",
                "anchor": "Hiring Update!",
                "date": "2025-01-02",
            },
            {"date": "2025-01-03", "description": "Untitled news"},
            {"title": "Hidden news", "anchor": "hidden-news", "draft": True},
        ],
        papers=[
            {"title": "Example Report", "date": "2025-01-01", "link": REPORT_URL},
            {
                "title": "Hidden report",
                "anchor": "hidden-report",
                "draft": True,
            },
        ],
        projects=[
            {"title": "Example Project"},
            {"title": "Custom project", "anchor": "custom-project"},
            {"title": "Hidden project", "anchor": "hidden-project", "draft": True},
        ],
    )

    for anchor in [
        "news-team-news",
        "hiring-update",
        "news-2025-01-03",
        "paper-example-report",
        "project-example-0",
        "custom-project",
    ]:
        assert homepage.count(f'id="{anchor}"') == 1
        assert homepage.count(f'href="#{anchor}"') == 1
    for anchor in ["hidden-news", "hidden-report", "hidden-project"]:
        assert f'id="{anchor}"' not in homepage
        assert f'href="#{anchor}"' not in homepage
    assert "<code>Custom news</code>" in homepage
    assert "<code>Example Project</code>" in homepage


@pytest.mark.parametrize(
    "page_url",
    [
        "/",
        "/blog/2025/03/debug-gym/",
        "/blog/2025/10/bug-pilot/",
        "/blog/2025/10/gistify/",
        "/blog/2026/06/shadow-frog/",
        "/blog/2026/08/negative-pi/",
        "/blog/2026/09/programdistill/",
    ],
)
def test_arxiv_links_open_pdfs(page_url):
    page = render_page(page_url)
    links = re.findall(r'href="(https://arxiv\.org/[^"]+)"', page)

    assert links
    assert all(link.startswith("https://arxiv.org/pdf/") for link in links), links


def test_programdistill_uses_arxiv_and_keeps_its_interactive_assets():
    page = render_page("/blog/2026/09/programdistill/")

    assert 'href="https://arxiv.org/pdf/2609.18805"' in page
    assert "ProgramDistill_arxiv.pdf" not in page
    assert 'id="programdistill-explorer"' in page
    for attribute, path in [
        ("src", "static/js/programdistill-explorer.js"),
        ("href", "static/css/programdistill-explorer.css"),
        ("data-src", "figures/programdistill/explorer/cases.json"),
        ("src", "static/images/programdistill-demo.mp4"),
    ]:
        assert f'{attribute}="/debug-gym/{path}' in page
        assert (DOCS / path).is_file()


def test_programdistill_preserves_the_shared_pdf_url():
    homepage = render_page()

    assert 'href="https://arxiv.org/pdf/2609.18805"' in homepage
    assert "ProgramDistill_arxiv.pdf" not in homepage
    pdf = DOCS / "static/papers/ProgramDistill_arxiv.pdf"
    assert pdf.is_file()
    assert pdf.read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize("baseurl", ["", "/debug-gym", "/preview"])
def test_programdistill_resource_buttons(baseurl):
    post_path = "/blog/2026/09/programdistill/"
    dashboard_path = post_path + "dashboard/"
    homepage = render_page(baseurl=baseurl)
    cards = re.split(r'<div class="project-card"[^>]*>', homepage)[1:]
    card = next(card for card in cards if f'href="{baseurl}{post_path}"' in card)
    post = render_page(post_path, baseurl=baseurl)
    hero = post.split('<section class="blog-hero">', 1)[1].split("</section>", 1)[0]

    for region in (card, hero):
        buttons = [
            anchor
            for anchor in re.findall(r"<a\b[^>]*>.*?</a>", region, re.S)
            if "<span>Leaderboard</span>" in anchor
        ]
        assert len(buttons) == 1
        assert f'href="{baseurl}{dashboard_path}"' in buttons[0]
        assert 'href="https://arxiv.org/pdf/2609.18805"' in region

    assert "<span>Read Blog Post</span>" in card
    assert "<span>Paper</span>" in card
    assert "<span>View paper</span>" in hero
    assert homepage.count("<span>Leaderboard</span>") == 1

    dashboard = render_page(dashboard_path, baseurl=baseurl)
    assert "data-pd-dashboard" in dashboard
    dashboard_hero = dashboard.split('<section class="blog-hero">', 1)[1].split(
        "</section>", 1
    )[0]
    for region in (card, hero, dashboard_hero):
        buttons = [
            anchor
            for anchor in re.findall(r"<a\b[^>]*>.*?</a>", region, re.S)
            if "<span>Dataset</span>" in anchor
        ]
        assert len(buttons) == 1
        assert (
            'href="https://huggingface.co/datasets/microsoft/ProgramDistill"'
            in buttons[0]
        )
        assert 'target="_blank"' in buttons[0]
        assert 'rel="noopener noreferrer"' in buttons[0]
    assert homepage.count("<span>Dataset</span>") == 1


@pytest.mark.parametrize(
    "page_url",
    ["/blog/2025/10/bug-pilot/", "/blog/2026/08/negative-pi/"],
)
def test_blog_headers_without_a_dataset_hide_the_button(page_url):
    page = render_page(page_url)
    hero = page.split('<section class="blog-hero">', 1)[1].split("</section>", 1)[0]

    assert "<span>Dataset</span>" not in hero


@pytest.mark.parametrize(
    "page_url",
    [
        "/blog/2025/10/bug-pilot/",
        "/blog/2026/08/negative-pi/",
        "/blog/2026/09/programdistill/dashboard/",
    ],
)
def test_blog_headers_without_a_leaderboard_link_are_unchanged(page_url):
    page = render_page(page_url)
    hero = page.split('<section class="blog-hero">', 1)[1].split("</section>", 1)[0]

    assert "<span>Leaderboard</span>" not in hero


def test_homepage_contact_section_uses_the_team_address():
    homepage = render_page()
    contact = re.search(
        r'<section class="[^"]*\bsection-contact\b[^"]*">(.*?)</section>',
        homepage,
        re.S,
    )

    assert contact is not None
    assert "Interested in collaborating or learning more" in contact.group(1)
    assert "<code>froggy@microsoft.com</code>" in contact.group(1)
    assert "debug-gym@microsoft.com" not in homepage
