"""
COMPONENT : Evaluation - golden question set
PURPOSE   : The 4 sample questions pipeline.txt phase 8 checks end to end
            with ask(): answer length, at least one citation, no
            "I don't know" fallback.
INPUTS    : -
OUTPUTS   : GOLDEN_QUESTIONS
DEPENDS ON: -
PANEL NOTE: pipeline.txt's brief included NVIDIA and Tesla; when this set was
            written only Apple and Microsoft were indexed (NVIDIA and IBM were
            added 2026-10-29 - their questions are in eval/cases.jsonl E96-E111). Its two
            in-scope samples are kept verbatim; the other two are replaced by
            Apple/Microsoft questions of the same kind (a metric lookup and a
            person query). `expect` lists figures/names verified on the PDF
            pages, so the golden check also tests correctness, not just shape.
"""

GOLDEN_QUESTIONS = [
    {"question": "Compare Apple and Microsoft revenue",                # pipeline.txt sample
     "expect": ["416,161", "281,724"]},                                # FY2025 revenue (Apple p32, Microsoft p69)
    {"question": "What risks does Apple mention?",                     # pipeline.txt sample
     "expect": []},                                                    # checked: >= 2 risk categories
    {"question": "What was Apple's net income in fiscal 2025?",
     "expect": ["112,010"]},                                           # Apple 10-K FY2025 p32
    {"question": "Who was Microsoft's chief executive officer in fiscal 2025?",
     "expect": ["Satya Nadella"]},                                     # Microsoft 10-K FY2025, executive officers
]
