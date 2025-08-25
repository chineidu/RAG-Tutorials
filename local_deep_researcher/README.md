<!-- markdownlint-disable MD033 -->

# Local Deep Research Project

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

**`Local Deep Researcher`** is a self-hosted, open-source AI research assistant that runs on your local hardware using Ollama or LM Studio.

It conducts deep, iterative web and document searches to produce detailed, cited reports while ensuring user privacy.

## Architecture

[![image.png](https://i.postimg.cc/2yNfQ2TW/image.png)](https://postimg.cc/LYD7Rkc4)

<br>

## Features

- **Graph-Based State Management:** Modular graph states for flexible workflow design.
- **Custom Configuration:** Easily adapt settings for different research scenarios.
- **Prompt Engineering:** Centralized prompt management for consistent LLM interactions.
- **Utility Functions:** Helper utilities for data processing and workflow automation.
- **Docker Support:** Containerized deployment for reproducibility and scalability.

## Directory Structure

- `local_deep_research/`
  - `graph_states.py`: Defines the states and transitions for the research graph.
  - `configuration.py`: Handles configuration management and environment setup.
  - `graph.py`: Main graph logic and orchestration.
  - `prompts.py`: Stores and manages prompt templates.
  - `utils.py`: Utility functions for data handling and processing.
  - `state.py`: State definitions and management.
- `docker-compose.yaml`: Docker configuration for running the project locally.

## Getting Started

1. **Clone the repository:**

   ```sh
   git clone <repo-url>
   cd RAG-Tutorials
   ```

2. **Configure your environment:**
   - Edit `local_deep_research/configuration.py` and `settings.py` as needed.
3. **Run with Docker:**

   ```sh
   docker-compose up
   ```

4. **Customize Graphs:**
   - Modify `graph_states.py` and `graph.py` to adapt workflows.

## Usage

- Copy and update the environment variables.

```sh
# Copy the example environment file
cp .env.example .env

# Launch LangGraph Server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev
```

- TBD

## License

This project is licensed under the MIT License.
