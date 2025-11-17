# Changelog

All notable changes to this project for the upcoming v1.0.0 release.

## Unreleased (pre-1.0.0)

- Add docstrings and inline comments across `src/` for clarity.
- Removed unused imports in `src/pose/pose_estimator.py` and `src/utils/angle_calculator.py`.
- Implement lazy Kalman filter initialization (`_init_filter`) in `src/filters/kalman_smoother.py`.
- Add safety check for `deproject()` return value in `src/run.py` to prevent unpacking `None`.
- Centralized MediaPipe landmark IDs into `src/utils/landmarks.py` and updated `run.py` and `angle_calculator.py` to use it.
- Restored `src/utils/angle_plotter.py` from build artifacts into the source tree for consistency.
- Updated `.gitignore` to exclude `build/`, `.venv/`, `dist/`, `*.egg-info/`, and editor settings.

## Notes and recommendations before release

- Replace `print()` debugging with `logging` for better control over verbosity.
- Add unit tests for `AngleCalculator`, `get_mean_depth`, and Kalman smoothing behaviour.
- Run linters (flake8/black/mypy) and set up CI to enforce style and tests.
