import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sources_parse():
    for path in (ROOT / "aether_2b").glob("*.py"):
        ast.parse(path.read_text())
    for path in (ROOT / "aether_pipeline").glob("*.py"):
        ast.parse(path.read_text())
    for path in (ROOT / "aether_kernels").glob("*.py"):
        ast.parse(path.read_text())
    ast.parse((ROOT / "train_end_to_end.py").read_text())
    ast.parse((ROOT / "scripts" / "smoke_pipeline.py").read_text())
    ast.parse((ROOT / "scripts" / "build_cuda_kernels.py").read_text())
    ast.parse((ROOT / "scripts" / "benchmark_cuda_kernels.py").read_text())


def test_pdf_text_was_extracted():
    text = (ROOT / "Aether.txt").read_text()
    assert "Manifold-Constrained Hyper-Connections" in text
    assert "Compressed Sparse Attention" in text
    assert "Muon Optimizer" in text


def test_technical_report_exists():
    report = (ROOT / "TECHNICAL_REPORT.md").read_text()
    assert "Aether tokenizer" in report
    assert "reference repo" in report
