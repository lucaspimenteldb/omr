"""
Testes de leitura do motor OMR.

Cobrem os 3 cenários (única / branco / múltipla) e a não-regressão em duas
imagens de natureza bem diferente:
  - samples/gabarito_matematica.png : scan achatado e reto (fácil)
  - test_images/foto_real.jpg       : foto de celular real, com perspectiva,
                                      sombra e marcas rabiscadas (difícil)

Rode com:  .venv/bin/python -m pytest -q
"""
import os

import pytest

from omr import engine

ROOT = os.path.dirname(os.path.dirname(__file__))


def _answers(rel_path):
    res = engine.read_image_file(os.path.join(ROOT, rel_path))
    return {r["question"]: (r["status"], r["answer"]) for r in res["results"]}


def test_scan_referencia_16_de_16():
    """O scan de referência: 16 marcações únicas, todas OK."""
    a = _answers("samples/gabarito_matematica.png")
    esperado = list("DBAADDBC" "CBBBABCD")
    assert len(a) == 16
    for q, letra in enumerate(esperado, start=1):
        assert a[q] == ("OK", letra), f"Q{q} veio {a[q]}, esperava OK/{letra}"


def test_foto_celular_real():
    """Foto de celular real: Q1 é dupla (A+D) => MULTIPLE; Q2-16 únicas."""
    a = _answers("test_images/foto_real.jpg")
    assert a[1][0] == "MULTIPLE", f"Q1 veio {a[1]}, esperava MULTIPLE (A e D)"
    esperado = {2: "B", 3: "A", 4: "A", 5: "D", 6: "D", 7: "B", 8: "C",
                9: "C", 10: "B", 11: "B", 12: "B", 13: "A", 14: "B",
                15: "C", 16: "D"}
    for q, letra in esperado.items():
        assert a[q] == ("OK", letra), f"Q{q} veio {a[q]}, esperava OK/{letra}"


@pytest.mark.parametrize("rel_path, questao, status_esperado", [
    ("test_images/test_blank.png", 1, "BLANK"),        # marca apagada
    ("test_images/test_multiple.png", 2, "MULTIPLE"),  # 2ª marca adicionada
])
def test_cenarios_sinteticos(rel_path, questao, status_esperado):
    """Cenários branco/múltipla (gerados por make_test_images.py)."""
    full = os.path.join(ROOT, rel_path)
    if not os.path.exists(full):
        pytest.skip(f"{rel_path} ausente — rode make_test_images.py para gerar")
    a = _answers(rel_path)
    assert a[questao][0] == status_esperado
