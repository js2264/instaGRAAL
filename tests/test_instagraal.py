"""instaGRAAL testing

Basic testing for the instaGRAAL scaffolder.
"""

import pathlib
import sys

import pytest
from unittest.mock import MagicMock, patch

import instagraal.instagraal as instagraal_module

EXAMPLE_DIR = pathlib.Path(__file__).parent.parent / "example"


def test_main_uses_correct_cycles(tmp_path):
    """Test that the instagraal main command runs with 5 cycles."""
    with patch.object(instagraal_module, "instagraal_class") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        mock_instance.simulation.level.S_o_A_frags = {"circ": 0}

        sys.argv = [
            "instagraal",
            str(EXAMPLE_DIR),
            str(EXAMPLE_DIR / "example.fa"),
            str(tmp_path),
            "--cycles=5",
        ]

        instagraal_module.main()

        mock_instance.full_em.assert_called_once_with(
            n_cycles=5,
            n_neighbours=5,
            bomb=False,
            id_start_sample_param=4,
            save_matrix=False,
        )
