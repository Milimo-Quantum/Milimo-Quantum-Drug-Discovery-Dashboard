# config

import json
import logging
from pydantic import BaseModel
from typing import List, Optional
import requests
from requests.exceptions import RequestException, Timeout
import numpy as np
import logging
import re
import sys

OLLAMA_BASE_URL = "http://localhost:11434/api"

def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger

class DrugModel(BaseModel):
    id: str
    name: str
    description: str
    molecular_formula: str
    molecular_weight: float
    quantum_params: dict

class ProteinModel(BaseModel):
    id: str
    name: str
    pdb_id: str
    description: str
    binding_site_residues: List[str]
    molecular_weight: float
    isoelectric_point: float

class SimulationResults(BaseModel):
    vqe_energy: float
    expectation_values: List[float]
    optimized_params: List[float]
    counts: dict
    quantum_time: float

class AnalysisResults(BaseModel):
    candidates: str
    affinity_improvement: str
    time_reduction: str
    adme_prediction: str
    quantum_state_distribution: str
    molecular_docking_visualization: str
    quantum_circuit: str

class SimulationRequest(BaseModel):
    drugId: str
    numQubits: int
    depth: int
    errorMitigation: str
    optimizationMethod: str
    ollamaModel: Optional[str] = None

class SimulationResponse(BaseModel):
    counts: dict
    quantum_time: float
    analysis: AnalysisResults
    circuit: str
    docking_visualization: str
    state_distribution_chart: str
    pipeline_visualization: str
    num_qubits: int
    depth: int
    vqe_energy: float

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an AI assistant specialized in quantum computing and drug discovery, 
working with the Milimo Quantum Drug Discovery Dashboard. Your role is to analyze 
quantum simulation results and provide insights for drug discovery processes. 
Please provide concise, numeric responses for each category when analyzing 
simulation results.
"""

def prepare_simulation_data(sim_results):
    """Prepare simulation data for CrewAI analysis by converting numpy arrays to lists and adding drug data."""
    prepared_data = {}
    for key, value in sim_results.items():
        if isinstance(value, np.ndarray):
            prepared_data[key] = value.tolist()
        elif isinstance(value, np.integer):
            prepared_data[key] = int(value)
        elif isinstance(value, np.floating):
            prepared_data[key] = float(value)
        else:
            prepared_data[key] = value
    
    return prepared_data

def is_ollama_server_running():
    try:
        logger.debug("Checking if Ollama server is running...")
        response = requests.get("http://localhost:11434/api/tags", timeout=600)
        logger.debug(f"Received response from Ollama server. Status code: {response.status_code}")
        return response.status_code == 200
    except RequestException as e:
        logger.error(f"Error checking Ollama server: {e}", exc_info=True)
        return False

def is_model_loaded(model_name):
    try:
        logger.debug(f"Checking if model {model_name} is loaded...")
        response = requests.get("http://localhost:11434/api/tags", timeout=600)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model in models:
                if model['name'] == model_name:
                    logger.debug(f"Model {model_name} is loaded")
                    return True
        logger.debug(f"Model {model_name} is not loaded")
        return False
    except RequestException as e:
        logger.error(f"Error checking if model is loaded: {e}", exc_info=True)
        return False

def call_ollama_api(prompt, model):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": f"{SYSTEM_PROMPT}\n\nHuman: {prompt}\n\nAssistant:",
        "stream": False
    }

    try:
        logger.debug(f"Sending request to Ollama API: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=120)
        logger.debug(f"Received response from Ollama API. Status code: {response.status_code}")
        response.raise_for_status()
        result = response.json().get('response', '')
        
        logger.debug(f"Full AI response:\n{result}")  # Log the full AI response
        
        # Parse the AI analysis result
        analysis = parse_ai_response(result)
        return analysis
    except Timeout:
        logger.error("Timeout error when calling Ollama API", exc_info=True)
        return generate_fallback_analysis()
    except RequestException as e:
        logger.error(f"Error calling Ollama API: {e}", exc_info=True)
        return generate_fallback_analysis()

def parse_ai_response(response):
    try:
        analysis = {
            "candidates": "Unable to determine",
            "affinity_improvement": "Unable to calculate",
            "time_reduction": "Unable to estimate",
            "adme_prediction": "Analysis unavailable",
            "quantum_state_distribution": "No analysis provided",
            "molecular_docking_visualization": "No analysis provided",
            "quantum_circuit": "No analysis provided"
        }

        # Split the response into lines for easier processing
        lines = response.split('\n')

        # Process each line
        for i, line in enumerate(lines):
            line = line.strip()
            
            if "Potential drug candidates:" in line:
                match = re.search(r'Potential drug candidates:\s*(\d+(?:-\d+)?)', line)
                if match:
                    analysis["candidates"] = match.group(1)
            
            elif "Binding affinity improvement:" in line:
                match = re.search(r'Binding affinity improvement:\s*([-]?\d+(?:\.\d+)?%?\s*(?:to\s*[-]?\d+(?:\.\d+)?%)?)', line)
                if match:
                    analysis["affinity_improvement"] = match.group(1)
            
            elif "Estimated time reduction in drug discovery:" in line:
                match = re.search(r'Estimated time reduction in drug discovery:\s*(\d+(?:-\d+)?)\s*months', line)
                if match:
                    analysis["time_reduction"] = f"{match.group(1)} months"
            
            elif "ADME prediction:" in line:
                adme_prediction = line.split("ADME prediction:", 1)[1].strip()
                analysis["adme_prediction"] = adme_prediction

            elif "**Quantum State Distribution**" in line:
                analysis["quantum_state_distribution"] = extract_section(lines[i+1:], "Quantum State Distribution")
            
            elif "**Molecular Docking Visualization**" in line:
                analysis["molecular_docking_visualization"] = extract_section(lines[i+1:], "Molecular Docking Visualization")
            
            elif "**Quantum Circuit**" in line:
                analysis["quantum_circuit"] = extract_section(lines[i+1:], "Quantum Circuit")

        return analysis

    except Exception as e:
        logger.error(f"Error parsing AI response: {e}", exc_info=True)
        logger.debug(f"AI response: {response}")
        return generate_fallback_analysis()

def extract_section(lines, section_name):
    section_content = []
    for line in lines:
        if line.strip().startswith("**"):
            break
        section_content.append(line.strip())
    return "\n".join(section_content).strip()

def generate_fallback_analysis():
    return {
        "candidates": "Unable to determine",
        "affinity_improvement": "Unable to calculate",
        "time_reduction": "Unable to estimate",
        "adme_prediction": "Analysis unavailable",
        "quantum_state_distribution": "Analysis not available due to error",
        "molecular_docking_visualization": "Analysis not available due to error",
        "quantum_circuit": "Analysis not available due to error"
    }

def check_ollama_server():
    if not is_ollama_server_running():
        print("Warning: Ollama server is not running. AI analysis will use fallback data.")
        print("To enable AI analysis, please start the Ollama server and restart this application.")
        user_input = input("Do you want to continue without AI analysis? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting application. Please start the Ollama server and try again.")
            sys.exit(1)

def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()['models']
            return [model['name'] for model in models]
        else:
            logger.error(f"Error fetching models: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return []