"""Unit tests never call real LLM/embedding APIs, but Settings requires the
keys to be present, so provide placeholders when they aren't set."""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-placeholder")
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")
