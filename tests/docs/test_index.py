import json
import subprocess
from html import unescape
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
REPORT_TITLE = "FrogNano: Training a 4B Coding Agent via Online Task Synthesis"
REPORT_PATH = "/static/papers/frognano_technical_report.pdf"
REPORT_DESCRIPTION = (
    "We introduce FrogNano, a 4B coding agent post-trained exclusively with "
    "reinforcement learning on synthetic software engineering tasks. Adapting "
    "tasks to the agent's evolving capabilities enables competitive performance "
    "without distillation from larger models."
)


def render_homepage(**options):
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
puts site.pages.find { |page| page.url == "/" }.output
""",
        ],
        cwd=DOCS,
        input=json.dumps(options),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("baseurl", ["", "/debug-gym", "/preview"])
def test_report_card_links_to_hosted_pdf(baseurl):
    homepage = render_homepage(baseurl=baseurl)
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
    assert f'href="{baseurl}{REPORT_PATH}"' in card
    assert (DOCS / REPORT_PATH.lstrip("/")).read_bytes().startswith(b"%PDF-")
    assert f'href="{baseurl}/blog/2026/08/negative-pi/"' in homepage
    assert "Apply Now" not in homepage


def test_report_links_and_feed_ordering():
    arxiv_url = "https://arxiv.org/abs/example"
    homepage = render_homepage(
        papers=[
            {
                "title": "Older report",
                "date": "2020-01-01",
                "link": REPORT_PATH,
            },
            {
                "title": "Draft report",
                "date": "2022-01-01",
                "link": REPORT_PATH,
                "draft": True,
            },
            {
                "title": "Newer report",
                "date": "2021-01-01",
                "link": REPORT_PATH,
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
    homepage = render_homepage(papers=[])

    assert "content-type-paper" not in homepage
    assert 'href="/debug-gym/blog/2026/08/negative-pi/"' in homepage
    assert "No updates yet" not in homepage
