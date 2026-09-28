from startshift.cli import build_parser


def test_cli_has_full_pipeline_commands():
    help_text = build_parser().format_help()
    for command in [
        "manifest",
        "make-splits",
        "make-targeted",
        "train",
        "baseline",
        "eval",
        "eval-id",
        "external-eval",
        "failure-template",
        "apply-failures",
        "audit",
        "gate",
        "aggregate",
        "report",
        "state-coverage",
    ]:
        assert command in help_text
