import json
import re
import subprocess
from html import unescape
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
REPORT_TITLE = "FrogNano: Training a 4B Coding Agent via Online Task Synthesis"
REPORT_URL = "https://arxiv.org/pdf/2609.07925"
LOCAL_REPORT_PATH = "/static/papers/ProgramDistill_arxiv.pdf"
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
def test_report_card_links_to_arxiv_pdf(baseurl):
    homepage = render_page(baseurl=baseurl)
    cards = homepage.split('<div class="project-card">')[1:]
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
    assert card.count("<a ") == 1
    assert f'href="{REPORT_URL}"' in card
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
    assert (DOCS / LOCAL_REPORT_PATH.lstrip("/")).read_bytes().startswith(b"%PDF-")


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
    cards = homepage.split('<div class="project-card">')[1:]

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


@pytest.mark.parametrize(
    "page_url",
    [
        "/",
        "/blog/2025/03/debug-gym/",
        "/blog/2025/10/bug-pilot/",
        "/blog/2025/10/gistify/",
        "/blog/2026/06/shadow-frog/",
        "/blog/2026/08/negative-pi/",
    ],
)
def test_arxiv_links_open_pdfs(page_url):
    page = render_page(page_url)
    links = re.findall(r'href="(https://arxiv\.org/[^"]+)"', page)

    assert links
    assert all(link.startswith("https://arxiv.org/pdf/") for link in links), links
