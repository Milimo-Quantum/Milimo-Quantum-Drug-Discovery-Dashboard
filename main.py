# main.py

from fasthtml.common import *
from api_handlers import setup_routes
from config import setup_logging
from visualization import create_drug_discovery_pipeline_visualization

# Initialize the FastHTML application with necessary headers
app, rt = fast_app(
    hdrs=[
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        Script(src="https://unpkg.com/htmx.org@1.8.4"),
        Script(src="https://unpkg.com/3dmol/build/3Dmol-min.js"),
        Style("""
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
            body { font-family: 'Roboto', sans-serif; background-color: #0f172a; color: #e2e8f0; }
            h1, h2, h3 { font-family: 'Orbitron', sans-serif; }
            .glassmorphism { 
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            }
            .quantum-title {
                background: linear-gradient(45deg, #60a5fa, #3b82f6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                filter: drop-shadow(0px 2px 2px rgba(59, 130, 246, 0.3));
            }
            .custom-scrollbar::-webkit-scrollbar {
                width: 8px;
            }
            .custom-scrollbar::-webkit-scrollbar-track {
                background: #1e293b;
            }
            .custom-scrollbar::-webkit-scrollbar-thumb {
                background-color: #4b5563;
                border-radius: 4px;
            }
            .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                background-color: #6b7280;
            }
            .loading-spinner {
                border: 4px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 4px solid #3b82f6;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .step-box {
                background-color: rgba(59, 130, 246, 0.1);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 16px;
            }
            .step-number {
                background-color: #3b82f6;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-right: 8px;
            }
            .chat-box {
                height: 300px;
                overflow-y: auto;
            }
            .user-message { color: #60a5fa; }
            .ai-message { color: #34d399; }
            .htmx-indicator{
                display:none;
            }
            .htmx-request .htmx-indicator{
                display:inline;
            }
            .htmx-request.htmx-indicator{
                display:inline;
            }
            #pdb-info p {
                color: #e2e8f0;
            }
            .input-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .input-field {
                flex-grow: 1;
                margin-right: 10px;
            }
            .input-value {
                min-width: 30px;
                text-align: right;
            }
        """)
    ]
)

setup_logging()
setup_routes(app, rt)

@rt("/")
def home():
    return Titled("Milimo Quantum Drug Discovery Dashboard"), Main(
        Div(id="particles-js", klass="fixed top-0 left-0 w-full h-full z-0 pointer-events-none"),
        Div(
            H1("Milimo Quantum AI-Enhanced Drug Discovery", klass="text-5xl font-bold mb-8 text-center quantum-title"),
            P("Accelerating pharmaceutical breakthroughs with quantum computing and AI", klass="text-2xl mb-12 text-center text-blue-200"),

            # User Guide
            Div(
                H2("How to Use This Dashboard", klass="text-3xl font-bold mb-6 text-blue-300"),
                Div(
                    Div(Span("1", klass="step-number"), "Upload a PDB or MOL2 file", klass="mb-2"),
                    Div(Span("2", klass="step-number"), "Adjust quantum parameters (number of qubits, circuit depth)", klass="mb-2"),
                    Div(Span("3", klass="step-number"), "Choose error mitigation and optimization methods", klass="mb-2"),
                    Div(Span("4", klass="step-number"), "Select an Ollama model for AI analysis", klass="mb-2"),
                    Div(Span("5", klass="step-number"), "Click 'Run Quantum Simulation' to start", klass="mb-2"),
                    Div(Span("6", klass="step-number"), "Explore the results in the visualizations below", klass="mb-2"),
                    Div(Span("7", klass="step-number"), "Chat with AI about the analysis results", klass="mb-2"),
                    klass="step-box text-blue-100"
                ),
                klass="mb-12 glassmorphism p-6"
            ),

            # Simulation Form
            Div(
                H2("Quantum Molecular Simulation", klass="text-3xl font-bold mb-6 text-blue-300"),
                P("Upload a PDB or MOL2 file and adjust parameters to simulate quantum molecular behavior:", klass="text-blue-100 mb-6"),
                P("This simulation uses quantum computing principles to model molecular interactions at the quantum level, potentially uncovering new drug candidates or optimizing existing ones.", klass="text-blue-100 mb-6"),
                Form(
                    Div(
                        Label("Upload PDB or MOL2 File: ", For="pdb-file", klass="block mb-2 text-blue-100"),
                        Input(type="file", id="pdb-file", name="pdbFile", accept=".pdb,.mol2", klass="w-full mb-6 p-2 bg-gray-800 text-white rounded",
                              hx_post="/upload_pdb",
                              hx_target="#pdb-info",
                              hx_swap="innerHTML",
                              hx_encoding="multipart/form-data"),
                        Div(id="pdb-info"),
                        klass="mb-6"
                    ),
                    Div(
                        Div(
                            Label("Number of Qubits: ", For="num_qubits", klass="block mb-2 text-blue-100"),
                            Input(type="number", min="2", max="10", value="5", id="num_qubits", name="numQubits", klass="w-full p-2 bg-gray-800 text-white rounded",
                                  placeholder="Enter the number of qubits"),
                            P("Qubits are the fundamental units of quantum information. More qubits allow for more complex simulations but are also more challenging to control.", klass="mt-2 text-sm text-blue-200"),
                            klass="mb-6"
                        ),
                        Div(
                            Label("Circuit Depth: ", For="depth", klass="block mb-2 text-blue-100"),
                            Input(type="number", min="1", max="10", value="3", id="depth", name="depth", klass="w-full p-2 bg-gray-800 text-white rounded",
                                  placeholder="Enter the circuit depth"),
                            P("Circuit depth represents the number of sequential quantum operations. Higher depth can model more complex interactions but is more susceptible to noise.", klass="mt-2 text-sm text-blue-200"),
                            klass="mb-6"
                        ),
                        id="quantum-parameters"
                    ),
                    Div(
                        Label("Error Mitigation: ", For="error_mitigation", klass="block mb-2 text-blue-100"),
                        Select(
                            Option("None", value="none"),
                            Option("Readout Error Mitigation", value="readout"),
                            Option("Zero Noise Extrapolation", value="zne"),
                            id="error_mitigation",
                            name="errorMitigation",
                            klass="w-full p-2 bg-gray-800 text-white rounded"
                        ),
                        P("Error mitigation techniques help reduce the impact of quantum noise on our results, improving the accuracy of our simulations.", klass="mt-2 text-sm text-blue-200"),
                        klass="mb-6"
                    ),
                    Div(
                        Label("Optimization Method: ", For="optimization_method", klass="block mb-2 text-blue-100"),
                        Select(
                            Option("SPSA", value="spsa"),
                            Option("COBYLA", value="cobyla"),
                            Option("ADAM", value="adam"),
                            id="optimization_method",
                            name="optimizationMethod",
                            klass="w-full p-2 bg-gray-800 text-white rounded"
                        ),
                        P("These are algorithms used to optimize the quantum circuit parameters. Different methods may perform better for different problems.", klass="mt-2 text-sm text-blue-200"),
                        klass="mb-6"
                    ),
                    Div(
                        H3("Ollama Model Selection", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Select(
                            id="ollama-model",
                            name="ollamaModel",
                            klass="w-full p-2 bg-gray-800 text-white rounded",
                            hx_get="/get_ollama_models",
                            hx_target="#ollama-model",
                            hx_trigger="load"
                        ),
                        klass="mb-12 glassmorphism p-6"
                    ),
                    Button("Run Quantum Simulation", type="submit", klass="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-lg mb-6 transition duration-300"),
                    hx_post="/run_simulation",
                    hx_target="#simulation-results",
                    hx_swap="innerHTML",
                    hx_indicator="#loading-indicator",
                    method="post",
                    enctype="multipart/form-data"
                ),
                # Loading Indicator
                Div(
                    Div(klass="loading-spinner"),
                    id="loading-indicator",
                    klass="htmx-indicator"
                ),
                klass="mb-12 glassmorphism p-6"
            ),

            # Simulation Results Placeholder
            Div(
                # This will be replaced by the simulation results
                id="simulation-results",
                klass="max-w-7xl mx-auto"
            ),

            klass="min-h-screen bg-gradient-to-b from-gray-900 to-blue-900 custom-scrollbar"
        )
    )

if __name__ == "__main__":
    serve()