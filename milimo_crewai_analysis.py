# milimo_crewai_analysis.py

import os
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from typing import Dict, Any, List
import MDAnalysis as mda

class MilimoCrewAIAnalysis:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
        self.llm = Ollama(model=ollama_model, base_url="http://localhost:11434")

    def create_agents(self):
        shared_instructions = """
        As you conduct your analysis, adhere to these comprehensive guidelines:
        1. Provide meticulously structured, publication-quality output.
        2. Utilize precise scientific language and cutting-edge terminology appropriate for experts in the field.
        3. Incorporate relevant quantitative data, statistical analysis, and error estimates where applicable.
        4. Highlight novel findings, unexpected results, or potential paradigm shifts that may significantly impact the field.
        5. Suggest specific, high-impact avenues for further research based on your analysis.
        6. Consider the broader implications of your analysis for personalized medicine, drug resistance, and emerging therapeutic modalities.
        7. Ensure all insights are actionable and directly relevant to advancing the drug discovery process.
        8. Format your output as plain text without any special formatting characters or markdown syntax.
        9. Use clear headings and subheadings to structure your report, but do not use asterisks or other symbols for emphasis.
        """

        pdb_structure_analyst = Agent(
            role="PDB Structure Analyst",
            goal="Analyze the uploaded PDB structure to provide insights on protein structure, potential binding sites, and implications for drug discovery.",
            backstory="You are a world-renowned expert in protein structure analysis, with a specific focus on identifying druggable targets and potential binding sites. Your insights have led to the discovery of numerous successful drugs.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            instructions=shared_instructions
        )

        quantum_analyst = Agent(
            role="Quantum Simulation and Analysis Expert",
            goal="Interpret quantum simulation data in the context of the PDB structure, providing insights on molecular interactions and quantum effects relevant to drug discovery.",
            backstory="You are a pioneer in quantum biology, specializing in applying quantum computing to molecular simulations. Your work has revolutionized our understanding of drug-target interactions at the quantum level.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            instructions=shared_instructions
        )

        molecular_dynamics_expert = Agent(
            role="Molecular Dynamics and Docking Expert",
            goal="Analyze the PDB structure and quantum simulation results to predict binding affinities, molecular dynamics, and potential drug-target interactions.",
            backstory="You are a leading authority in computational chemistry, specializing in molecular dynamics simulations and protein-ligand docking. Your algorithms have significantly improved the accuracy of virtual screening in drug discovery.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            instructions=shared_instructions
        )

        drug_strategist = Agent(
            role="Drug Discovery Strategist",
            goal="Evaluate the PDB structure and quantum-enhanced insights to develop comprehensive drug discovery strategies and assess development potential.",
            backstory="You are a visionary pharmaceutical researcher with decades of experience in drug development strategies. Your unique approach combines traditional pharmacology with cutting-edge AI and quantum computing techniques.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            instructions=shared_instructions
        )

        adme_expert = Agent(
            role="ADME Prediction Specialist",
            goal="Predict ADME properties based on the PDB structure and quantum simulation results, offering insights into the pharmacokinetic profile of potential drug candidates.",
            backstory="You are a pioneer in integrating structural biology and quantum chemistry into ADME predictions. Your innovative approaches have significantly improved early-stage drug candidate assessment.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[],
            instructions=shared_instructions
        )

        return [pdb_structure_analyst, quantum_analyst, molecular_dynamics_expert, drug_strategist, adme_expert]

    def create_tasks(self, simulation_data: Dict[str, Any], file_path: str) -> List[Task]:
        agents = self.create_agents()

        # Parse the molecular file for additional information
        universe = mda.Universe(file_path)
        atoms = universe.atoms

        # Extract relevant information
        num_atoms = len(atoms)
        num_residues = len(universe.residues)
        chains = list(set(res.segid for res in universe.residues))
        if not chains or chains == ['']:
            chains = ['N/A']

        tasks = [
            Task(
                description=f"""Analyze the molecular structure and provide insights:
                1. File Path: {file_path}
                2. Number of atoms: {num_atoms}
                3. Number of residues: {num_residues}
                4. Chains: {', '.join(chains)}

                Provide a comprehensive report addressing:
                - The overall structure of the molecule and its potential as a drug target
                - Identification of potential binding sites or druggable pockets
                - Analysis of key structural features that may impact drug binding
                - Implications of the structure for drug discovery strategies

                Include quantitative assessments and comparisons to known drug targets where applicable.""",
                agent=agents[0],  # PDB Structure Analyst
                expected_output="A detailed analysis report on the molecular structure."
            ),
            Task(
                description=f"""Interpret the quantum simulation results in the context of the molecular structure:
                1. VQE energy: {simulation_data['vqe_energy']}
                2. Quantum state distribution: {simulation_data['counts']}
                3. Optimized parameters: {simulation_data['optimized_params']}
                4. Number of qubits: {simulation_data['num_qubits']}
                5. Circuit depth: {simulation_data['depth']}
                6. Error mitigation strategy: {simulation_data['error_mitigation']}

                Provide a detailed analysis addressing:
                - How the quantum simulation results relate to the molecular structure
                - Insights into molecular stability and potential quantum effects in drug-target interactions
                - Interpretation of the optimized parameters in the context of the molecular structure
                - Recommendations for improving the quantum simulation based on the molecular data

                Include visualizations and quantitative comparisons where applicable.""",
                agent=agents[1],  # Quantum Analyst
                expected_output="An in-depth interpretation of the quantum simulation results in relation to the molecular structure."
            ),
            Task(
                description=f"""Conduct a molecular dynamics and docking analysis based on the molecular structure and quantum simulation results:
                1. Molecular structure information (atoms: {num_atoms}, residues: {num_residues}, chains: {', '.join(chains)})
                2. Quantum simulation results (VQE energy: {simulation_data['vqe_energy']}, optimized parameters: {simulation_data['optimized_params']})

                Provide a comprehensive report addressing:
                - Prediction of binding affinities and potential drug-target interactions
                - Analysis of molecular flexibility and its impact on drug binding
                - Identification of key residues or atoms for drug interactions based on the structure and quantum results
                - Suggestions for potential lead compounds or modifications to existing compounds

                Include molecular dynamics predictions and docking scores where applicable.""",
                agent=agents[2],  # Molecular Dynamics Expert
                expected_output="A comprehensive report on molecular dynamics and docking analysis."
            ),
            Task(
                description=f"""Develop a quantum-enhanced drug discovery strategy based on the molecular structure and simulation results:
                1. Molecular structure insights (provided by the PDB Structure Analyst)
                2. Quantum simulation insights (provided by the Quantum Analyst)
                3. Molecular dynamics and docking analysis (provided by the Molecular Dynamics Expert)

                Create a comprehensive drug discovery strategy addressing:
                - Evaluation of the target's druggability based on structural and quantum insights
                - Proposed screening strategies incorporating quantum and structural information
                - Suggestions for structure-based drug design approaches
                - Timeline and resource estimates for the drug discovery process
                - Potential challenges and mitigation strategies

                Provide a detailed roadmap with specific, actionable steps for the drug discovery process.""",
                agent=agents[3],  # Drug Strategist
                expected_output="A detailed drug discovery strategy incorporating quantum and structural insights."
            ),
            Task(
                description=f"""Predict ADME properties based on the molecular structure and quantum simulation results:
                1. Molecular structure information (atoms: {num_atoms}, residues: {num_residues}, chains: {', '.join(chains)})
                2. Quantum simulation results (VQE energy: {simulation_data['vqe_energy']}, optimized parameters: {simulation_data['optimized_params']})

                Provide detailed ADME predictions addressing:
                - How the molecular structure and quantum results might influence ADME properties
                - Absorption predictions based on structural features
                - Distribution estimates considering molecule interactions
                - Metabolism predictions focusing on potential metabolic sites
                - Excretion predictions based on physicochemical properties derived from the structure

                Include quantitative predictions and comparisons to known drugs targeting similar structures.""",
                agent=agents[4],  # ADME Expert
                expected_output="A detailed report on predicted ADME properties based on the provided data."
            ),
        ]

        return tasks

    def run_analysis(self, simulation_data: Dict[str, Any], file_path: str) -> str:
        tasks = self.create_tasks(simulation_data, file_path)

        crew = Crew(
            agents=self.create_agents(),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            manager_llm=self.llm
        )

        result = crew.kickoff()
        
        # Save each agent's output to a separate file
        os.makedirs("agent_outputs", exist_ok=True)
        for i, task in enumerate(tasks):
            with open(f"agent_outputs/agent_{i+1}_output.txt", "w") as f:
                f.write(task.output.raw)
        
        return result.raw

def run_crewai_analysis(simulation_data: Dict[str, Any], ollama_model: str, file_path: str) -> str:
    crewai_system = MilimoCrewAIAnalysis(ollama_model)
    return crewai_system.run_analysis(simulation_data, file_path)