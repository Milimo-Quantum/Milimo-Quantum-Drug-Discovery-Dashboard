# Milimo Quantum Drug Discovery Dashboard

## Description

The Milimo Quantum Drug Discovery Dashboard is an innovative web application that leverages quantum computing and artificial intelligence to accelerate the drug discovery process. This cutting-edge tool combines quantum simulation techniques with AI-driven analysis to provide insights into molecular interactions and potential drug candidates.

## Key Features

- Quantum Molecular Simulation: Utilizes Qiskit to perform quantum simulations of molecular structures.
- AI-Enhanced Analysis: Employs advanced AI models to interpret quantum simulation results and provide insights.
- Interactive Visualizations: Offers visual representations of quantum states, molecular docking, and quantum circuits.
- PDB and MOL2 File Support: Allows users to upload and analyze PDB or MOL2 files for molecular structures.
- Customizable Quantum Parameters: Users can adjust quantum parameters such as the number of qubits and circuit depth.
- Real-time AI Chat: Enables users to interact with an AI assistant to discuss analysis results and ask questions.

## Technologies Used

- FastHTML: A Python-based web framework for building server-rendered hypermedia applications.
- Qiskit: An open-source framework for quantum computing.
- HTMX: A lightweight JavaScript library for AJAX, CSS Transitions, and WebSockets.
- Tailwind CSS: A utility-first CSS framework for rapid UI development.
- MDAnalysis: A Python library for analyzing molecular dynamics simulations.
- CrewAI: An AI framework for orchestrating role-playing, autonomous AI agents.
- Ollama: An open-source AI model serving platform.

## Setup and Installation

1. Clone the repository:
2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
3. Install the required dependencies: (pip install -r requirements.txt)
4. Set up Ollama:
- Install Ollama following the instructions at [Ollama's official website](https://ollama.ai/).
- Ensure the Ollama server is running before starting the application.

5. Run the application: (python main.py)
6. Open a web browser and navigate to `http://localhost:5000` to access the dashboard.

## Usage

1. Upload a PDB or MOL2 file of the molecular structure you want to analyze.
2. Adjust quantum parameters such as the number of qubits and circuit depth.
3. Select error mitigation and optimization methods.
4. Choose an Ollama model for AI analysis.
5. Click "Run Quantum Simulation" to start the analysis.
6. Explore the results in the visualizations provided.
7. Use the AI chat feature to ask questions about the analysis results.

## Contributing

Contributions to the Milimo Quantum Drug Discovery Dashboard are welcome! Please feel free to submit pull requests, create issues, or suggest new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Qiskit team for their excellent quantum computing framework.
- The FastHTML, HTMX, and Tailwind CSS communities for their powerful web development tools.
- The creators of MDAnalysis, CrewAI, and Ollama for their invaluable contributions to scientific computing and AI.
