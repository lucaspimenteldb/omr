"""
Entrada de imagem: bytes ou arquivo -> matriz BGR pronta para os fluxos.

Aceita o que o celular do usuário produzir — em especial **HEIC/HEIF**, o
formato padrão da câmera do iPhone, que o OpenCV não lê de jeito nenhum. Quem
decodifica é o Pillow (com o plugin `pillow-heif` para HEIC/HEIF; AVIF já vem
no Pillow 12); o OpenCV fica de rede de segurança. O porquê dessa ordem está
em `decodificar`.

Não há conversão para arquivo no meio do caminho: a foto vira matriz de pixels
direto. Salvar um PNG intermediário só gastaria tempo, e salvar um JPEG
perderia qualidade de verdade — o artefato de bloco do JPEG cai justamente na
borda da bolha, que é o que o motor mede.
"""
from __future__ import annotations

from io import BytesIO

import numpy as np

import cv2
from PIL import Image, ImageOps

from .registration import EntradaInvalida

try:  # HEIC/HEIF (iPhone). Opcional: sem ele, o resto continua funcionando.
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_DISPONIVEL = True
except ImportError:  # pragma: no cover - só acontece em instalação incompleta
    HEIF_DISPONIVEL = False

# Fotos de celular passam fácil dos 12 MP; acima disso o custo cresce sem
# ganho de leitura nenhum (a folha canônica tem 1700 px de largura).
LADO_MAXIMO = 2600

# Pillow recusa imagens gigantes por medo de zip bomb. Uma foto de 200 MP é
# implausível como gabarito, mas o limite padrão (~89 MP) é baixo para scanner
# de mesa em 600 dpi, então subimos com folga em vez de desligar a proteção.
Image.MAX_IMAGE_PIXELS = 300_000_000


def _reduzir(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    maior = max(h, w)
    if maior <= LADO_MAXIMO:
        return img
    esc = LADO_MAXIMO / float(maior)
    return cv2.resize(img, (int(round(w * esc)), int(round(h * esc))), interpolation=cv2.INTER_AREA)


def _rotulo_do_formato(data: bytes) -> str:
    """Nome do formato pelos bytes iniciais — só para a mensagem de erro.

    Vale a pena porque o erro genérico ("formato não suportado") não diz ao
    usuário o que fazer, e os dois casos que aparecem na prática — HEIC do
    iPhone e PDF — têm respostas bem diferentes.
    """
    if data[:5] == b"%PDF-":
        return "PDF"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        marca = data[8:12]
        if marca[:2] in (b"he", b"mi", b"ms"):
            return "HEIC"
        if marca[:2] == b"av":
            return "AVIF"
    return ""


def _para_bgr(im: Image.Image) -> np.ndarray:
    """PIL -> BGR 8 bits, respeitando EXIF e achatando transparência."""
    # O cv2.imdecode gira pelo EXIF; o Pillow não. Sem isto, a mesma foto sairia
    # deitada num caminho e em pé no outro.
    #
    # Em HEIC isto é de propósito um no-op: o iPhone grava o giro na caixa
    # `irot` do próprio HEIF, o libheif já aplica na decodificação, e o
    # pillow-heif zera a tag EXIF para ninguém girar de novo (as_plugin.py,
    # set_orientation). NÃO troque isto por `info["original_orientation"]`:
    # medido em 82 fotos de iPhone, 59 trazem esse campo com valor 6 e já vêm
    # em pé — honrá-lo deitaria todas elas.
    im = ImageOps.exif_transpose(im)

    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        # Scanner de 16 bits: converter direto para RGB trunca e estoura tudo.
        a = np.asarray(im, dtype=np.float32)
        faixa = float(a.max()) - float(a.min())
        a = (a - a.min()) * (255.0 / faixa) if faixa > 0 else np.zeros_like(a)
        return cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")

    if im.mode in ("RGBA", "LA"):
        # Papel transparente é papel branco. Sem achatar, o canal RGB por baixo
        # costuma vir preto e o motor leria a folha inteira como tinta.
        fundo = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(fundo, im.convert("RGBA"))

    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _decodificar_com_pillow(data: bytes) -> np.ndarray | None:
    try:
        with Image.open(BytesIO(data)) as im:
            return _para_bgr(im)
    except Exception:  # formato desconhecido, arquivo truncado, etc.
        return None


def decodificar(image_bytes: bytes) -> np.ndarray:
    """Bytes de qualquer formato de imagem suportado -> BGR.

    O Pillow decodifica; o OpenCV fica de rede de segurança para arquivo que o
    Pillow recusar (JPEG truncado, por exemplo). A ordem é essa de propósito:

      - só o Pillow lê HEIC/HEIF, e é o formato padrão da câmera do iPhone;
      - `cv2.IMREAD_COLOR` descarta o canal alfa, e o RGB embaixo de pixel
        transparente costuma ser preto — uma folha exportada com fundo
        transparente seria lida como papel inteiro pintado;
      - `IMREAD_UNCHANGED`, que preservaria o alfa, ignora a orientação EXIF;
      - num JPEG de 12 MP a diferença de tempo medida foi de 6% (102 ms contra
        96 ms), irrelevante perto do registro da folha.
    """
    if not image_bytes:
        raise EntradaInvalida("Arquivo vazio.")

    img = _decodificar_com_pillow(image_bytes)
    if img is None:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise EntradaInvalida(_recado_de_formato(image_bytes))
    return _reduzir(img)


def _recado_de_formato(data: bytes) -> str:
    formato = _rotulo_do_formato(data)
    if formato == "PDF":
        return ("Isso é um PDF, não uma imagem. Exporte a página como foto "
                "(JPG ou PNG) e envie de novo.")
    if formato == "HEIC" and not HEIF_DISPONIVEL:
        return ("Foto HEIC (iPhone), mas o servidor está sem o suporte a HEIC. "
                "Instale a dependência `pillow-heif` ou envie a foto como JPG.")
    if formato:
        return f"Não consegui decodificar a imagem ({formato} corrompido ou incompleto?)."
    return ("Não consegui decodificar a imagem. Envie uma foto em JPG, PNG ou "
            "HEIC.")


def ler_arquivo(path: str) -> np.ndarray:
    """Lê do disco pelo mesmo caminho do upload — mesmos formatos, mesmos erros."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise EntradaInvalida(f"Não consegui abrir a imagem: {path} ({e.strerror})") from None
    try:
        return decodificar(data)
    except EntradaInvalida as e:
        raise EntradaInvalida(f"{path}: {e}") from None


def ler_bytes(image_bytes: bytes, fluxo: str, debug: bool = False):
    """Atalho: bytes -> resultado do fluxo pedido."""
    from .flows import ler_fluxo

    return ler_fluxo(fluxo, decodificar(image_bytes), debug=debug)
