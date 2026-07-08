"""The nbmock module provides support for mocking datasets to speed up testing."""

from os import environ


def mock_datasets():
    """Mock the datasets to use the tiny versions for testing."""
    import unittest.mock as mock
    from capymoa.datasets import ElectricityTiny, CovtypeTiny, FriedTiny
    from capymoa.ocl.datasets import TinySplitMNIST

    mock.patch("capymoa.datasets.Electricity", ElectricityTiny).start()
    mock.patch("capymoa.datasets.Covtype", CovtypeTiny).start()
    mock.patch("capymoa.datasets.Fried", FriedTiny).start()
    mock.patch("capymoa.ocl.datasets.SplitMNIST", TinySplitMNIST).start()


def override_prequential_evaluation(max_instances: int = 100):
    """Monkeypatch the prequential evaluation to limit the number of instances.

    This is useful for testing purposes to speed up the evaluation.
    """
    import capymoa.evaluation as evaluation
    from capymoa.evaluation import prequential_evaluation as _prequential_evaluation

    def prequential_evaluation(*args, **kwargs):
        kwargs["max_instances"] = max_instances
        return _prequential_evaluation(*args, **kwargs)

    evaluation.prequential_evaluation = prequential_evaluation


def is_nb_fast() -> bool:
    """Should the notebook be run with faster settings.

    Some notebooks are slow to run because they use large datasets and run
    for many iterations. This is good for documentation purposes but not for
    testing. This function returns True if the notebook should be run with
    faster settings.

    Care should be taken to hide cells in capymoa.org that are meant for testing
    only. This is done by adding ``"nbsphinx": "hidden"`` to the cell metadata.
    See: https://nbsphinx.readthedocs.io/en/0.9.3/hidden-cells.html
    """
    return bool(environ.get("NB_FAST", False))
