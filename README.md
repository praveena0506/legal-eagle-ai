Legal Eagle AI is an End-to-End MLOps project designed to predict the outcome of Indian Supreme Court cases ("Allowed" vs. "Dismissed") based on case text.

Unlike static ML projects, this system features a live, self-healing data pipeline. It automatically scrapes real-time legal news, auto-labels the data using a custom NLP rule engine, and retrains the PyTorch model to stay current with the latest judicial trends.

🏗️ Architecture
The system follows a modern Microservices architecture:

Code snippet

graph LR
    A[Google News RSS] -->|Scraper| B(Apache Airflow DAG)
    B -->|Raw Text| C{Auto-Labeler Engine}
    C -->|Labeled Data| D[(MongoDB Atlas Cloud)]
    D -->|Training Data| E[PyTorch Trainer]
    E -->|Model Weights| F[FastAPI Inference Engine]
    F -->|JSON Prediction| G[Client/API Consumer]
🛠️ Tech Stack
Language: Python 3.12

Dependency Management: Poetry

API Framework: FastAPI (Asynchronous)

ML Core: PyTorch (Custom Neural Network)

Database: MongoDB Atlas (Cloud NoSQL)

Orchestration: Apache Airflow (Scheduled Pipelines)

Containerization: Docker

✨ Key Features
1. 🌊 Automated Data Pipeline (ETL)
Ingestion: Scrapes real-time legal updates from Google News RSS feeds using BeautifulSoup and lxml.

Orchestration: Designed to use Apache Airflow to schedule daily fetch jobs, ensuring the dataset never goes stale.

Cloud Storage: All data is pushed immediately to a MongoDB Atlas cluster, preventing data loss in ephemeral environments (like Heroku).

2. 🧹 The "Vacuum Cleaner" Auto-Labeler
Implements a custom NLP heuristics engine that cleans and labels raw news data.

Logic: Scans for keywords (e.g., "Acquitted", "Conviction Set Aside" → WIN vs. "Dismissed", "Upheld" → LOSS).

Result: Converts raw unstructured text into labeled training data without human intervention.

3. 🧠 Self-Learning AI Model
A custom text classification model built with PyTorch.

Uses a SimpleTokenizer and Embedding Bag architecture for efficient, low-latency inference.

Achieved a significant loss reduction (1.08 → 0.26) during initial training cycles.

4. 🐳 Dockerized Deployment
Fully containerized application using Dockerfile.

Uses Poetry to lock dependencies, ensuring the environment is reproducible across Dev, Test, and Prod stages.

⚙️ Installation & Setup
Prerequisites:

Docker Desktop

Python 3.10+

Poetry

1. Clone the Repository

Bash

git clone https://github.com/yourusername/legal-eagle-ai.git
cd legal-eagle-ai
2. Install Dependencies (via Poetry)

Bash

poetry install
3. Configure Environment Create a .env file or export your MongoDB URI:

Bash

export MONGO_URI="mongodb+srv://<user>:<pass>@cluster0.mongodb.net/?appName=Cluster0"
4. Run with Docker

Bash

docker build -t legal-eagle-api .
docker run -p 8000:8000 legal-eagle-api
🔮 Future Roadmap
[ ] Implement BERT embeddings for deeper legal context understanding.

[ ] Deploy Airflow DAGs to a managed instance (e.g., Astronomer).

[ ] Add CI/CD pipelines using GitHub Actions.