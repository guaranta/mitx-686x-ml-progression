# mitx-686x-ml-progression

**MITx 6.86x — Machine Learning with Python: From Linear Models to Deep Learning**

Progressão intencional do MicroMasters SDSC: classificadores lineares com margem → visão computacional → aprendizado não supervisionado (EM / mistura gaussiana).

---

## Objetivos de estudo

O 6.86x não ensina ML como caixa-preta — ensina uma **progressão de complexidade controlada**. Este repositório reproduz três marcos: **(1)** classificação linear com perceptron, average perceptron e Pegasos (regularização + SGD); **(2)** escalonamento para MNIST com modelos lineares e MLP; **(3)** aprendizado não supervisionado com EM para completar matrizes (estilo Netflix). O objetivo é internalizar que redes profundas são extensão de ideias lineares com não-linearidades e regularização — não magia.

---

## Resultados — Project 1 (sentiment)

| Classificador | Accuracy | Hinge loss |
|---------------|----------|------------|
| Perceptron | **0.865** | 0.6331 |
| Average Perceptron | **0.915** | 0.2882 |
| Pegasos (λ=0.2) | **0.920** | 0.2756 |

![Comparação accuracy e hinge loss](docs/figures/sentiment_comparison.png)

O gráfico à esquerda mostra o salto de accuracy do perceptron básico (0.865) para average (0.915) e Pegasos (0.920). À direita, o hinge loss cai pela metade — average perceptron **estabiliza** o vetor de pesos ao longo das iterações, e Pegasos adiciona **regularização L2** que evita overfitting. A lição: em dados ruidosos, estabilização e regularização valem mais que mais épocas no perceptron vanilla.

---

## Progressão MNIST

![Linear → Softmax → MLP accuracy](docs/figures/mnist_progression.png)

A curva ascendente ilustra o ganho incremental: regressão linear (~82%) captura separabilidade parcial; softmax (~88%) modela multiclasse; MLP (~91%) adiciona não-linearidade. O gap entre linear e MLP quantifica **o valor da capacidade do modelo** — em produção, comece linear (interpretável, rápido) e escale complexidade só quando o ganho justifica custo de inferência.

---

## Módulos

| Módulo | Tópico | Comando |
|--------|--------|---------|
| `01_sentiment` | Perceptron, Avg, Pegasos | `python 01_sentiment/run.py` |
| `02_mnist` | Linear → MLP | `python 02_mnist/run.py` |
| `03_matrix_completion` | EM gaussiano | `python 03_matrix_completion/run.py` |

## Setup

```bash
pip install -r requirements.txt
python docs/generate_figures.py
```

---

## Aprendizados e aplicação no mercado

A progressão linear → neural → não supervisionado espelha decisões reais em AI Engineering: classificadores de **sentimento/intenção** em chatbots (Project 1), **visão computacional** em inspeção e OCR (Project 2), e **recomendação/completude** em sistemas de retrieval (Project 3). Pegasos prefigura SGD em LLMs; EM prefigura misturas gaussianas em detecção de anomalias. Para CTO, este repo demonstra que a stack de IA não começa em transformers — começa em **margens, perdas e convergência**, com complexidade adicionada apenas quando mensurável.

---

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
