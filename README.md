# Talent Scout

Sistema web para cadastro, análise e classificação de atletas com base em métricas físicas e de desempenho, utilizando um modelo de Machine Learning (KNN) para agrupá-los em níveis de talento.

## Visão geral

O sistema recebe medições de atletas (altura, envergadura, arremesso, salto horizontal, abdominais, sexo, idade) e:

- Classifica cada atleta em um dos clusters de desempenho: **Iniciante**, **Intermediário**, **Competitivo** ou **Elite**, usando um modelo K-Nearest Neighbors treinado com `scikit-learn`.
- Disponibiliza um dashboard web para cadastro, edição, filtragem e visualização estatística dos atletas (PCA, correlação, radar de perfil, métricas de qualidade do cluster).
- Permite importar/exportar dados em CSV e exportar relatórios (CSV + gráficos) em ZIP.

## Arquitetura

O projeto segue uma organização MVC:

```
src/
├── main.py            # Ponto de entrada / CLI
├── config.py          # Configuração do Flask, SQLAlchemy e colunas do domínio
├── controller/        # Rotas Flask (Controller)
├── model/
│   ├── model.py        # Fachada do model, orquestra os "elos"
│   ├── athleteModel.py # Entidade Athlete (SQLAlchemy)
│   ├── knnModel.py      # Modelo KNN de classificação
│   └── elos/            # Cadeia de responsabilidade para import/export/análise
├── view/
│   ├── templates/       # Templates Jinja2
│   └── static/          # CSS
└── utils/               # Geração de dados sintéticos, dataclasses e utilitários
```

Diagrama de classes: [`docs/classDiagram.mmd`](docs/classDiagram.mmd).
Protótipo de tela: [`docs/screenPrototype.html`](docs/screenPrototype.html).

## Requisitos

- Python 3.10+ (usa `match`/`case`)
- pip

## Instalação

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Crie um arquivo `.env` (opcional) para sobrescrever as configurações padrão:

```
SECRET_KEY=troque-esta-chave
DEBUG=True
HOST=0.0.0.0
PORT=5000
```

## Uso

O ponto de entrada é `src/main.py`, executado como módulo:

```bash
# Inicia o servidor web
python -m src.main

# Popula o banco de dados com atletas sintéticos
python -m src.main --populate 200

# Limpa o banco e popula com 160 atletas
python -m src.main --populate 160 --clear

# Exporta os dados do banco para CSV
python -m src.main --export dados.csv

# Exibe estatísticas sobre os dados no banco
python -m src.main --stats

# Gera dados sintéticos sem inserir no banco
python -m src.main --generate-data 100

# Combina qualquer comando acima sem subir o servidor ao final
python -m src.main --populate 200 --no-server
```

Após iniciar o servidor, acesse `http://localhost:5000` (ou o host/porta configurados).

### Páginas principais

| Rota          | Descrição                                              |
|---------------|---------------------------------------------------------|
| `/`           | Dashboard geral                                         |
| `/cadastro`   | Cadastro de atletas (formulário e importação de CSV)     |
| `/analise`    | Listagem com filtros, ordenação e paginação              |
| `/view`       | Visualizações estatísticas (PCA, correlação, radar etc.) |
| `/atleta/editar/<id>` | Edição de atleta                                  |
| `/exportData` | Exportação de dados (CSV + gráficos) em ZIP              |

## Modelo de dados

Um atleta (`Athlete`) possui: `nome`, `dataNascimento`, `sexo`, `altura`, `envergadura`, `arremesso`, `saltoHorizontal`, `abdominais` e `cluster` (nível calculado pelo modelo KNN).

## Stack

Flask · Flask-SQLAlchemy · SQLite · scikit-learn · pandas/numpy · matplotlib/seaborn · ReportLab (PDF) · joblib
