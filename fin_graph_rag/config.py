"""
COMPONENT : Configuration (UI choices)
PURPOSE   : Lists the UI offers: companies in scope and the answer models a user
            may pick. Secrets are not here - they stay in .env (loaded by
            fin_graph_rag.embedders.load_env).
INPUTS    : FGR_ANSWER_MODELS (optional, comma-separated) from the environment
OUTPUTS   : COMPANIES, ANSWER_MODELS
DEPENDS ON: -
PANEL NOTE: Only free-tier Gemini models are offered (project rule: free
            services only). Both were listed as available by the Gemini API on
            2026-10-28; Flash is larger and slower, Flash-Lite is the default.
"""

import os

# IBM and NVIDIA added 2026-10-29 (phases 1-5 run for them; see README).
COMPANIES = {"Apple": "apple", "Microsoft": "microsoft", "IBM": "ibm", "NVIDIA": "nvidia"}
ANSWER_MODELS = [m.strip() for m in os.getenv(
    "FGR_ANSWER_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash").split(",") if m.strip()]
