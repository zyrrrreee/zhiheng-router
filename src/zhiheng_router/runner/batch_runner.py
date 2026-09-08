"""Sequential execution of multiple Queries through BenchmarkRunner."""

from .benchmark_runner import BenchmarkRunError, BenchmarkRunner, RunRecord


class BatchRunner:
    def __init__(self, runner: BenchmarkRunner) -> None:
        if not isinstance(runner, BenchmarkRunner):
            raise TypeError("runner must be a BenchmarkRunner")
        self._runner = runner

    def run_many(self, queries: list[tuple[str, str]]) -> list[RunRecord]:
        if not isinstance(queries, list):
            raise TypeError("queries must be a list of (query_id, query_text) tuples")
        records: list[RunRecord] = []
        for item in queries:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each query must be a (query_id, query_text) tuple")
            query_id, query_text = item
            try:
                record = self._runner.run_one(query_id, query_text)
            except BenchmarkRunError as exc:
                record = exc.record
            records.append(record)
        return records
