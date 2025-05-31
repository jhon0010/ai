# AI Project

## Overview
This repository contains an AI project that utilizes LangChain for natural language processing and article generation. The project is set up with a specific Python environment to ensure consistent development and deployment.

## Prerequisites
- Python 3.12.7
- uv package manager

## Installation

### 1. Install uv Package Manager
First, install the uv package manager which is used for Python environment management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Up Python Environment
Navigate to the project root directory and execute the following commands:

```bash
# Install Python 3.12.7
uv python install 3.12.7

# Create virtual environment
uv venv --python 3.12.7

# Install dependencies
uv sync
```

## Usage
To run the LangChain article generation script:

```bash
python3 src/langchain/langchain_runner.py
```

## Environment Variables
The project requires the following environment variables to be set in a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `TAVILY_API_KEY`: Your Tavily API key
- `LANGSMITH_API_KEY`: Your LangSmith API key
- `LANGSMITH_TRACING`: Set to "true" for tracing
- `LANGSMITH_PROJECT`: Project name for LangSmith

## Project Structure

```
├── src/
│ └── langchain/
│   └── runner.py
├── .env
└── README.md
```