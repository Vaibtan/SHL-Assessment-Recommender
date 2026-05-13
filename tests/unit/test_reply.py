# Purpose: Unit tests for the assembly/reply layer.

from __future__ import annotations

from shl_recommender.assembly.reply import (
    assemble_chat_response,
    compute_end_of_conversation,
    render_shortlist_table,
)
from shl_recommender.schemas import Recommendation
from tests.unit.test_validator import _make_index


def test_compute_end_of_conversation_invariant() -> None:
    rec = Recommendation(name="A", url="https://www.shl.com/a", test_type="K")
    assert compute_end_of_conversation(False, [rec]) is False
    assert compute_end_of_conversation(True, []) is False
    assert compute_end_of_conversation(True, [rec]) is True


def test_render_shortlist_table_has_header_and_rows() -> None:
    idx = _make_index([("1", "Java 8 (New)", "K")])
    table = render_shortlist_table(idx.items)
    lines = table.splitlines()
    assert lines[0].startswith("| # | Name |")
    assert lines[1].startswith("|---|")
    assert "Java 8 (New)" in lines[2]
    assert "https://www.shl.com/1" in lines[2]


def test_assemble_chat_response_embeds_table_when_recs_present() -> None:
    idx = _make_index([("1", "A", "K"), ("2", "B", "P")])
    resp = assemble_chat_response(
        reply_text="Here's the shortlist.",
        entity_ids=["1", "2"],
        is_final_turn_flag=True,
        index=idx,
    )
    assert resp.end_of_conversation is True
    assert len(resp.recommendations) == 2
    assert "Here's the shortlist." in resp.reply
    assert "| # | Name |" in resp.reply


def test_assemble_chat_response_no_table_when_no_recs() -> None:
    idx = _make_index([("1", "A", "K")])
    resp = assemble_chat_response(
        reply_text="Ask me about a role.",
        entity_ids=[],
        is_final_turn_flag=True,  # invariant should still produce end=False
        index=idx,
    )
    assert resp.end_of_conversation is False
    assert "| # | Name |" not in resp.reply
    assert resp.reply == "Ask me about a role."


def test_assemble_drops_invalid_ids_silently() -> None:
    idx = _make_index([("1", "A", "K")])
    resp = assemble_chat_response(
        reply_text="x",
        entity_ids=["1", "999", "fake"],
        is_final_turn_flag=False,
        index=idx,
    )
    assert len(resp.recommendations) == 1
    assert resp.recommendations[0].url == "https://www.shl.com/1"
