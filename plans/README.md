# Planos de Implementação

Gerados pela skill `improve` em 2026-07-27. Projeto sem git — o "drift check" de
cada plano é feito comparando os excertos de "Estado atual" com o código vivo.

Contexto: o motor OMR lê bem um scan achatado, mas errava quase tudo numa foto
de celular real. A causa raiz foi **medida** (grade de amostragem com
coordenadas fixas que não sobrevivem à perspectiva da foto) e a correção do
Plano 001 já foi **prototipada e validada** contra a foto que falhou
(`test_images/foto_real.jpg`): passa de ~2/16 para 16/16, sem regressão no scan.

## Ordem de execução & status

| Plano | Título | Prioridade | Esforço | Depende de | Status |
|------|-------|-----------|--------|-----------|--------|
| 001 | Detecção de grade ancorada nas bolhas (+ normalização de iluminação) | P1 | M | — | FEITO (2026-07-27) |

> 001 aplicado em `omr/config.py` + `omr/engine.py`; testes em `tests/test_reading.py`
> (4 passando). Referência 16/16 e foto de celular 16/16 (Q1 = MULTIPLE real).

Valores de status: TODO | EM ANDAMENTO | FEITO | BLOQUEADO (motivo em 1 linha) | REJEITADO (motivo).

## Notas de dependência

- O Plano 001 depende de existir `test_images/foto_real.jpg` no repositório
  (a foto de celular de teste). Já está presente. Um conjunto de teste maior
  (achado #5) tornaria a validação muito mais forte — ver follow-ups.

## Achado escolhido

O usuário pediu foco na **causa raiz**. O Plano 001 ataca exatamente ela (o
alinhamento rígido). O achado #2 (iluminação) foi absorvido no 001 porque a
normalização flat-field é necessária para o pipeline alinhado ler correto na
foto real — estão acoplados na solução validada.

## Follow-ups deferidos (não pedidos agora, mas reais)

Priorizados por leverage; vire planos quando quiser:

- **#5 — Conjunto de teste real (P1, S).** Hoje a validação é 1 scan + 1 foto.
  Coletar 10–20 fotos (ângulos, luz, marcas a lápis/caneta/rabisco) com gabarito
  esperado anotado transforma qualquer mudança futura em algo mensurável. É o
  que dá confiança de que o 001 generaliza. Deveria vir logo após o 001.
- **#3 — Limiar relativo por questão (P2, M).** O corte absoluto (`MARK_THRESHOLD`)
  funcionou em tudo que testamos porque a separação é grande (~56 pontos), mas
  marcas a lápis fraco podem encostar na zona `REVIEW`. Decidir por comparação
  relativa entre as 4 alternativas é mais robusto para estilos de marca variados.
- **#4 — Detecção de box robusta (P2, M).** `_find_boxes` exige 4 cantos convexos
  e pode falhar com dedo na foto, curvatura do papel ou sombra forte na borda.
  Na foto de teste ela funcionou, mas é o próximo ponto frágil.
- **Caminho B — Marcadores fiduciais na folha (P2, plano de design).** A correção
  durável: quadradinhos pretos nos 4 cantos + registro por eles (padrão OMRChecker).
  Exige reimprimir o modelo, então é decisão de produto — plano de design à parte.

## Achados considerados e rejeitados

- "Trocar a razão de aspecto do canônico": rejeitado — os aspectos medidos são
  próximos (0.65 vs 0.66–0.68); não era a causa. A causa é a POSIÇÃO das bolhas
  dentro do box, resolvida detectando-as (Plano 001), não ajustando o aspecto.
- "Subir o `MARK_THRESHOLD` para matar os falsos 100%": rejeitado — os 100%
  falsos vinham de amostrar a BORDA do box por desalinhamento; mexer no limiar
  mascararia o sintoma sem corrigir a causa.
