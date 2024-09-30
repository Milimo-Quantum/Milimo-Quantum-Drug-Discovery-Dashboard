# api_handlers.py

from fasthtml.common import *
from quantum_simulation import run_quantum_simulation, circuit_to_image
from visualization import generate_molecular_docking_visualization, create_quantum_state_distribution_chart, create_drug_discovery_pipeline_visualization
from config import prepare_simulation_data, is_ollama_server_running, is_model_loaded, call_ollama_api, generate_fallback_analysis
from milimo_crewai_analysis import run_crewai_analysis
import logging
import requests
import json
from config import OLLAMA_BASE_URL, SimulationRequest, SimulationResponse
import os
from starlette.responses import StreamingResponse
import asyncio
import aiohttp
import tempfile
import base64
import numpy as np
import re

# Import MDAnalysis for processing PDB and MOL2 files
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction

logger = logging.getLogger(__name__)

def calculate_secondary_structure(universe):
    """A simple approximation of secondary structure based on dihedral angles"""
    ss_count = {"helix": 0, "sheet": 0, "coil": 0}
    protein = universe.select_atoms('protein')
    for ts in universe.trajectory[:1]:  # Only need one frame for static structures
        phi_angles = mda.lib.distances.calc_bonds(protein.select_atoms('backbone and name C'), protein.select_atoms('backbone and name N'))
        psi_angles = mda.lib.distances.calc_bonds(protein.select_atoms('backbone and name N'), protein.select_atoms('backbone and name C'))
        # Simple classification based on placeholder thresholds
        for phi, psi in zip(phi_angles, psi_angles):
            if -70 < phi < -30 and -70 < psi < -30:
                ss_count["helix"] += 1
            elif -150 < phi < -90 and 90 < psi < 150:
                ss_count["sheet"] += 1
            else:
                ss_count["coil"] += 1
    return ss_count

def setup_routes(app, rt):
    @rt("/upload_pdb", methods=["POST"])
    async def upload_pdb(req):
        logger.info("Entering upload_pdb function")
        form = await req.form()
        logger.info(f"Form data keys: {form.keys()}")
        pdb_file = form.get("pdbFile")
        logger.info(f"PDB file object type: {type(pdb_file)}")
        
        if not pdb_file:
            logger.warning("No file was uploaded")
            return P("No file was uploaded.", klass="text-red-500")
        
        try:
            if isinstance(pdb_file, str):
                logger.info("PDB file is a string, attempting to decode base64")
                if ',' in pdb_file:
                    content = base64.b64decode(pdb_file.split(',')[1])
                else:
                    content = base64.b64decode(pdb_file)
                filename = "uploaded.pdb"
            elif hasattr(pdb_file, 'filename'):
                logger.info(f"PDB file is a file object with filename: {pdb_file.filename}")
                content = await pdb_file.read()
                filename = pdb_file.filename
            else:
                logger.error(f"Uploaded file object is invalid. Type: {type(pdb_file)}")
                return P(f"Uploaded file object is invalid. Type: {type(pdb_file)}", klass="text-red-500")
            
            logger.info(f"Filename: {filename}")
            
            if not filename.lower().endswith(('.pdb', '.mol2')):
                logger.warning(f"Uploaded file is not a PDB or MOL2 file. Filename: {filename}")
                return P(f"Uploaded file is not a PDB or MOL2 file. Filename: {filename}", klass="text-red-500")
            
            if not content:
                logger.warning("Uploaded file is empty")
                return P("Uploaded file is empty.", klass="text-red-500")
            
            logger.info(f"Content length: {len(content)} bytes")
            
            # Create a temporary file to store the uploaded content
            if filename.lower().endswith('.pdb'):
                suffix = '.pdb'
            else:
                suffix = '.mol2'
            
            # Preprocess the MOL2 file if necessary
            if suffix == '.mol2':
                content = preprocess_mol2(content)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            logger.info(f"Temporary file created: {temp_file_path}")

            # Load the structure using MDAnalysis
            universe = mda.Universe(temp_file_path)
            atoms = universe.atoms

            # Extract relevant information
            num_atoms = len(atoms)
            num_residues = len(universe.residues)
            chains = list(set(res.segid for res in universe.residues))
            if not chains or chains == ['']:
                chains = ['N/A']

            # Calculate simple secondary structure
            ss_counts = calculate_secondary_structure(universe)
            ss_info = ", ".join([f"{ss}: {count}" for ss, count in ss_counts.items()])

            # Extract header information if available
            structure_method = 'unknown'
            resolution = 'Not specified'
            r_free = 'Not specified'
            r_work = 'Not specified'

            # Store the temporary file path in the session for later use
            req.session['temp_file_path'] = temp_file_path

            return Div(
                P(f"File: {filename}", klass="mb-2"),
                P(f"Number of atoms: {num_atoms}", klass="mb-2"),
                P(f"Number of residues: {num_residues}", klass="mb-2"),
                P(f"Chains: {', '.join(chains)}", klass="mb-2"),
                P(f"Structure determination method: {structure_method}", klass="mb-2"),
                P(f"Resolution: {resolution}", klass="mb-2"),
                P(f"R-free: {r_free}", klass="mb-2"),
                P(f"R-work: {r_work}", klass="mb-2"),
                P(f"Approximate Secondary Structure: {ss_info}", klass="mb-2"),
                klass="text-white"
            )
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return P(f"Error processing file: {str(e)}", klass="text-red-500")

    def preprocess_mol2(content):
        """Adjusts the MOL2 content to fix inconsistencies between NO_CHARGES and charge columns."""
        try:
            content_str = content.decode('utf-8')
            # Check if 'NO_CHARGES' is specified
            if 'NO_CHARGES' in content_str:
                # Remove the charge column from atom entries
                lines = content_str.splitlines()
                new_lines = []
                atom_section = False
                for line in lines:
                    if line.startswith('@<TRIPOS>ATOM'):
                        atom_section = True
                        new_lines.append(line)
                        continue
                    if line.startswith('@<TRIPOS>BOND'):
                        atom_section = False
                        new_lines.append(line)
                        continue
                    if atom_section and line.strip():
                        # Remove the last column (charge)
                        parts = line.split()
                        if len(parts) > 8:
                            line = ' '.join(parts[:-1])
                    new_lines.append(line)
                content_str = '\n'.join(new_lines)
            else:
                # If charges are present, ensure the header reflects that
                content_str = content_str.replace('NO_CHARGES', 'USER_CHARGES')
            return content_str.encode('utf-8')
        except Exception as e:
            logger.error(f"Error in preprocess_mol2: {str(e)}")
            return content  # Return original content if preprocessing fails

    @rt("/run_simulation", methods=["POST"])
    async def run_simulation(req):
        try:
            form_data = await req.form()
            temp_file_path = req.session.get('temp_file_path')
            if not temp_file_path:
                return Div(P("Please upload a PDB or MOL2 file before running the simulation.", klass="text-red-500"), id="simulation-results")

            num_qubits = int(form_data.get('numQubits'))
            depth = int(form_data.get('depth'))
            error_mitigation = form_data.get('errorMitigation')
            optimization_method = form_data.get('optimizationMethod')
            ollama_model = form_data.get('ollamaModel', None)

            # Run quantum simulation
            sim_results = run_quantum_simulation(temp_file_path, num_qubits, depth, error_mitigation, optimization_method)

            # Generate visualizations
            circuit_svg = circuit_to_image(sim_results['optimized_circuit'])
            docking_visualization = generate_molecular_docking_visualization(
                sim_results['vqe_result'],
                sim_results['expectation_values'],
                sim_results['optimized_params'],
                temp_file_path
            )
            state_distribution_chart = create_quantum_state_distribution_chart(sim_results['counts'])

            # Prepare simulation data for AI analysis
            simulation_data = prepare_simulation_data({
                "file_path": temp_file_path,
                "num_qubits": num_qubits,
                "depth": depth,
                "vqe_energy": float(sim_results['vqe_result'].optimal_value),
                "expectation_values": sim_results['expectation_values'],
                "error_mitigation": error_mitigation,
                "optimization_method": optimization_method,
                "counts": sim_results['counts'],
                "optimized_params": sim_results['optimized_params'],
                "quantum_time": sim_results['quantum_time'],
            })

            # Run AI analysis
            if ollama_model and is_ollama_server_running() and is_model_loaded(ollama_model):
                logger.info(f"Running AI analysis with model {ollama_model}")
                try:
                    ai_analysis = run_crewai_analysis(simulation_data, ollama_model, temp_file_path)
                except Exception as e:
                    logger.error(f"Error during AI analysis: {str(e)}")
                    ai_analysis = f"AI analysis failed: {str(e)}"
            else:
                logger.warning(f"Ollama server is not running, model not selected, or model {ollama_model} is not loaded. Using fallback analysis.")
                ai_analysis = "AI analysis unavailable. Please check Ollama server and model configuration."

            # Clean up the temporary file
            os.unlink(temp_file_path)
            del req.session['temp_file_path']

            # Import NotStr to include raw HTML content
            from fasthtml.common import NotStr

            # Build the content to return
            content = Div(
                # Simulation Results
                Div(
                    Div(
                        H3("Quantum State Distribution", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Div(
                            Img(src=state_distribution_chart, klass="w-full h-full object-contain"),
                            klass="h-80 glassmorphism",
                            id="simulation-chart"
                        ),
                        P("This chart displays the probability distribution of quantum states after simulation.", klass="mt-4 text-sm text-blue-100"),
                        klass="w-full mb-12"
                    ),
                    Div(
                        H3("Molecular Docking Visualization", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Div(
                            Img(src=docking_visualization, klass="w-full h-full object-contain"),
                            klass="h-80 glassmorphism",
                            id="docking-visualization"
                        ),
                        P("This visualization shows the predicted binding pose of the molecule with its target.", klass="mt-4 text-sm text-blue-100"),
                        klass="w-full mb-12"
                    ),
                    Div(
                        H3("Quantum Circuit", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Div(
                            Div(
                                NotStr(circuit_svg),
                                klass="flex justify-center",
                                id="quantum-circuit-svg"
                            ),
                            P(f"Qubits: {num_qubits}, Depth: {depth}", id="circuit-info", klass="text-center mt-4 text-blue-100"),
                            klass="glassmorphism p-6"
                        ),
                        P("This diagram represents the quantum circuit used in the simulation.", klass="mt-4 text-sm text-blue-100"),
                        klass="w-full mb-12"
                    ),
                    Div(
                        H3("AI Molecule Analysis", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Div(
                            Div(
                                P(ai_analysis, klass="text-blue-100 whitespace-pre-wrap"),
                                klass="text-left"
                            ),
                            klass="p-6 glassmorphism flex items-center justify-center",
                            id="analysis-result"
                        ),
                        P("The AI analysis provides insights on potential molecules.", klass="mt-4 text-sm text-blue-100"),
                        Button(
                            "Chat with AI about Analysis",
                            id="chat-button",
                            klass="mt-4 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded",
                            hx_get=f"/chat?model={ollama_model}&analysis={ai_analysis}",
                            hx_target="#chat-container",
                            hx_swap="innerHTML",
                            hx_trigger="click once"
                        ),
                        klass="w-full mb-12"
                    ),
                    Div(
                        id="chat-container",
                        klass="w-full mb-12"
                    ),
                    Div(
                        H3("Drug Discovery Pipeline", klass="text-2xl font-bold mb-4 text-blue-300"),
                        Div(
                            Img(src=create_drug_discovery_pipeline_visualization(), klass="w-full"),
                            klass="glassmorphism p-6",
                            id="drug-discovery-pipeline"
                        ),
                        P("Version 0.1.0", klass="mt-4 text-sm text-blue-100"),
                        klass="w-full mb-12"
                    ),
                    klass="max-w-7xl mx-auto"
                ),
                id="simulation-results"
            )

            # Return the content to be swapped into the #simulation-results div
            return content
        except Exception as e:
            logger.error(f"Error in run_simulation: {str(e)}", exc_info=True)
            return Div(P(f"An error occurred: {str(e)}", klass="text-red-500"), id="simulation-results")

    @rt("/chat")
    def chat(model: str, analysis: str):
        return Div(
            H3("Quantum Analysis Chat", klass="text-2xl font-bold mb-4 text-blue-300"),
            Div(
                id="chat-messages",
                klass="chat-box mb-4 p-4 glassmorphism h-80 overflow-y-auto"
            ),
            Form(
                Input(type="text", name="message", placeholder="Type your message here...", klass="w-full p-2 mb-2 bg-gray-800 text-white rounded", id="chat-input"),
                Input(type="hidden", name="model", value=model),
                Input(type="hidden", name="analysis", value=analysis),
                Button("Send", type="submit", klass="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"),
                hx_post="/chat-message",
                hx_target="#chat-messages",
                hx_swap="beforeend",
                hx_trigger="submit",
                hx_indicator="#loading-indicator"
            ),
            Div(
                "Loading...",
                id="loading-indicator",
                klass="htmx-indicator"
            ),
            klass="w-full mb-12 glassmorphism p-6"
        )

    @rt("/chat-message", methods=["POST"])
    async def chat_message(req):
        form_data = await req.form()
        user_message = form_data.get('message')
        model = form_data.get('model')
        analysis = form_data.get('analysis')
        
        # Load agent outputs
        agent_outputs = ""
        for i in range(1, 6):  # Assuming 5 agents
            try:
                with open(f"agent_outputs/agent_{i}_output.txt", "r") as f:
                    agent_outputs += f"Agent {i} Output:\n{f.read()}\n\n"
            except FileNotFoundError:
                logger.warning(f"Agent {i} output file not found")

        # Prepare the prompt with the analysis context and agent outputs
        prompt = f"""You are an AI assistant specialized in quantum computing and drug discovery. 
        You have access to the following analysis of a quantum simulation:

        {analysis}

        Additionally, you have access to the following agent outputs:

        {agent_outputs}

        Based on this information, please respond to the following user query:

        User: {user_message}

        Assistant:"""

        # Call Ollama API
        api_url = f"{OLLAMA_BASE_URL}/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        async def generate_response():
            yield f"""
            <div class="mb-2">
                <p class="user-message">You: {user_message}</p>
            </div>
            <div id="ai-response" class="ai-message mb-4">AI: """

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, json=payload) as response:
                        if response.status == 200:
                            async for line in response.content:
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        if 'message' in chunk and 'content' in chunk['message']:
                                            yield chunk['message']['content']
                                    except json.JSONDecodeError:
                                        continue  # Skip lines that can't be parsed as JSON
                        else:
                            yield f"Error: Received status code {response.status} from Ollama API"
            except Exception as e:
                yield f"Error: {str(e)}"

            yield """</div>
            <script>
                htmx.find('#chat-input').value = '';
                htmx.find('#chat-messages').scrollTop = htmx.find('#chat-messages').scrollHeight;
            </script>
            """

        return StreamingResponse(generate_response(), media_type="text/html")

    @rt("/get_ollama_models")
    def get_ollama_models():
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                options = [Option(model['name'], value=model['name']) for model in models]
                return options
            else:
                return Option("No models available", value="")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return Option("Error fetching models", value="")