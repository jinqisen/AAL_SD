import joblib
from joblib import Parallel, delayed


def test_run_single_experiment_forces_joblib_threading_backend(monkeypatch):
    import src.run_parallel_strict as run_parallel_strict

    observed = {}

    class DummyPipeline:
        def __init__(self, config, experiment_name, run_id=None):
            self.config = config
            self.experiment_name = experiment_name
            self.run_id = run_id

        def run_and_collect(self, log_path):
            backend, _ = joblib.parallel.get_active_backend()
            observed["backend"] = backend.__class__.__name__
            Parallel(n_jobs=2)(delayed(lambda x: x)(i) for i in range(2))

    monkeypatch.setattr(run_parallel_strict, "ActiveLearningPipeline", DummyPipeline)

    exp_name = "dummy_joblib_backend"
    run_id = "__joblib_backend_test__"
    result = run_parallel_strict.run_single_experiment(
        (exp_name, run_id, 0, False, None, None, None)
    )

    assert result[0] == exp_name
    assert result[1] == "success"
    assert observed.get("backend") == "ThreadingBackend"

