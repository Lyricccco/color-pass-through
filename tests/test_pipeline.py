import torch

from color_pass_through.calibration import generate_phi_grid
from color_pass_through.pipeline import apply_observer_correction


def test_phi_grid_has_125_unique_values() -> None:
    grid = generate_phi_grid()
    assert len(grid) == 125
    assert len(set(grid)) == 125
    assert grid[0] == (0.025, 0.025, 0.025)
    assert grid[-1] == (0.075, 0.075, 0.075)


def test_paper_correction_sign() -> None:
    camera = torch.full((1, 3, 2, 2), 0.5)
    coefficient = torch.full((1, 1, 2, 2), 2.0)
    corrected = apply_observer_correction(
        camera, coefficient, (0.1, 0.05, 0.025), correction_sign=-1
    )
    expected = torch.tensor([0.3, 0.4, 0.45]).view(1, 3, 1, 1).expand_as(camera)
    torch.testing.assert_close(corrected, expected)
