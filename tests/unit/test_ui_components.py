"""Unit tests for ui_components.py."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ui_components import (
    DummyViewer,
    get_full_leaderboard_data,
)


class TestGetFullLeaderboardData:
    """Tests for get_full_leaderboard_data function."""

    @patch("ui_components.get_leaderboard_viewer_instance")
    def test_returns_empty_for_dummy_viewer(self, mock_get_viewer):
        """When DummyViewer is returned (error case), should return empty DataFrame."""
        # DummyViewer is created when data loading fails
        error_df = pd.DataFrame({"Message": ["Error loading data"]})
        dummy_viewer = DummyViewer(error_df)
        mock_get_viewer.return_value = (dummy_viewer, {"Overall": []})

        df, tag_map = get_full_leaderboard_data("test")

        assert df.empty
        assert tag_map == {}

    @patch("ui_components.get_leaderboard_viewer_instance")
    def test_returns_empty_for_empty_leaderboard_viewer(self, mock_get_viewer):
        """When LeaderboardViewer has no data, should return empty DataFrame."""
        from agenteval.leaderboard.view import LeaderboardViewer

        mock_viewer = MagicMock(spec=LeaderboardViewer)
        mock_viewer._load.return_value = (pd.DataFrame(), {})
        mock_get_viewer.return_value = (mock_viewer, {"Overall": []})

        df, tag_map = get_full_leaderboard_data("test")

        assert df.empty
        assert tag_map == {}

    @patch("ui_components.transform_raw_dataframe")
    @patch("ui_components.create_pretty_tag_map")
    @patch("ui_components.get_leaderboard_viewer_instance")
    def test_transforms_valid_data(self, mock_get_viewer, mock_create_tag_map, mock_transform):
        """When LeaderboardViewer has valid data, should transform and return it."""
        from agenteval.leaderboard.view import LeaderboardViewer

        # Create mock viewer with valid data
        mock_viewer = MagicMock(spec=LeaderboardViewer)
        raw_df = pd.DataFrame({
            "agent_name": ["Test Agent"],
            "score": [0.5],
        })
        mock_viewer._load.return_value = (raw_df, {})

        raw_tag_map = {"overall": ["benchmark1"]}
        mock_get_viewer.return_value = (mock_viewer, raw_tag_map)

        # Mock the transformation functions
        transformed_df = pd.DataFrame({
            "Agent": ["Test Agent"],
            "Score": ["50.0%"],
        })
        mock_transform.return_value = transformed_df

        pretty_tag_map = {"Overall": ["Benchmark 1"]}
        mock_create_tag_map.return_value = pretty_tag_map

        df, tag_map = get_full_leaderboard_data("test")

        assert not df.empty
        assert "Agent" in df.columns
        mock_transform.assert_called_once_with(raw_df)
        mock_create_tag_map.assert_called_once()

    @patch("ui_components.get_leaderboard_viewer_instance")
    def test_returns_empty_for_unexpected_type(self, mock_get_viewer):
        """When an unexpected type is returned, should return empty DataFrame."""
        # This shouldn't happen in practice, but tests the fallback
        mock_get_viewer.return_value = ("unexpected_string", {})

        df, tag_map = get_full_leaderboard_data("test")

        assert df.empty
        assert tag_map == {}


class TestDummyViewer:
    """Tests for DummyViewer class."""

    def test_load_returns_error_dataframe(self):
        """DummyViewer._load() should return the error DataFrame it was initialized with."""
        error_df = pd.DataFrame({"Message": ["Test error message"]})
        dummy = DummyViewer(error_df)

        result_df, result_map = dummy._load()

        assert result_df.equals(error_df)
        assert result_map == {}
