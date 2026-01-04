from crewai import Agent, Task, Crew, Process

import os
from .mock_llm import MockLLM

class CrewManager:
    def __init__(self):
        # If USE_STUB_LLM is truthy, operate in dev stub mode (no LLMs required)
        self.stub_mode = os.getenv("USE_STUB_LLM", "0").lower() in ("1", "true", "yes")
        if self.stub_mode:
            # Skip creating real Agents — they'll be simulated at runtime
            self.pii_officer = None
            self.analyst = None
            self.gatekeeper = None
            return

        # 1. Define Agents (Roles & Backstories)
        self.pii_officer = self._create_agent(
            role='Privacy Enforcement Officer',
            goal='Mask user metadata and ensure data privacy',
            backstory='Expert in GDPR and data anonymization. You ensure no PII enters the analysis stream.',
            verbose=True,
            allow_delegation=False
        )

        self.analyst = self._create_agent(
            role='Health Content Researcher',
            goal='Analyze health claims for factual accuracy',
            backstory='A specialized medical fact-checker trained to spot misinformation patterns in text and images.',
            verbose=True
        )

        self.gatekeeper = self._create_agent(
            role='Similarity & XAI Gatekeeper',
            goal='Provide final explainable verdict based on all evidence',
            backstory='The final decision maker who synthesizes multimodal data into a human-readable explanation.',
            verbose=True
        )

    def _create_agent(self, **kwargs):
        try:
            return Agent(**kwargs)
        except (ImportError, ValueError, RuntimeError) as e:
            # Provide actionable guidance for missing provider configuration (e.g., OPENAI_API_KEY)
            raise RuntimeError(
                "Failed to create Agent due to missing or invalid LLM configuration. "
                "Make sure the environment variable OPENAI_API_KEY is set before starting the server. "
                "In PowerShell (temporary): $Env:OPENAI_API_KEY='your-key' ; In CMD: set OPENAI_API_KEY=your-key ; To persist: setx OPENAI_API_KEY 'your-key'."
            ) from e

    def execute_verification(self, tweet_text, image_url, masked_user_id):
        # 2. Define Tasks
        analysis_task = Task(
            description=f"Analyze the following content: Text: {tweet_text}, Image: {image_url}. Determine if it is Health Misinformation.",
            expected_output="A score between 0 and 1 and a brief factual reason.",
            agent=self.analyst
        )

        final_gate_task = Task(
            description=f"Based on the analysis, provide a final verdict for user {masked_user_id}. Explain WHY the decision was made.",
            expected_output="Final Label (REAL/FAKE), Score, and an Explainable AI (XAI) Reason.",
            agent=self.gatekeeper,
            context=[analysis_task] # This is where the 'Gatekeeper' waits for the 'Analyst'
        )

        # 3. Assemble the Crew
        # If running in stub mode, return a deterministic mock response
        if self.stub_mode:
            return self._stub_verification(tweet_text, image_url, masked_user_id)

        acas_crew = Crew(
            agents=[self.pii_officer, self.analyst, self.gatekeeper],
            tasks=[analysis_task, final_gate_task],
            process=Process.sequential # Ensures Linear Pipeline: Analysis -> Gatekeeper
        )

        return acas_crew.kickoff()

    def _stub_verification(self, tweet_text, image_url, masked_user_id):
        """Simulate a multi-agent run using a deterministic MockLLM.

        Steps simulated:
        - Analyst: analyze text and optional image
        - Similarity: compare against a tiny local corpus
        - Gatekeeper: synthesize analyst + similarity into final verdict
        """
        mock = MockLLM()

        # Analyst analysis
        analyst_result = mock.analyze(tweet_text, image_url=image_url)

        # Simulated similarity check against a small, local corpus of example documents
        local_corpus = [
            "Official health study shows vaccine efficacy in randomized trial",
            "This is an unproven claim with no peer-reviewed evidence",
            "Public health guidance from CDC"
        ]
        sim_result = mock.compare_similarity(tweet_text, existing_corpus=local_corpus)

        # Convert similarity into an analysis-like dict so it participates in scoring
        similarity_analysis = {
            "label": "REAL" if sim_result.get("similarity", 0.0) >= 0.5 else "FAKE",
            "score": sim_result.get("similarity", 0.0),
            "reason": f"Similarity={sim_result.get('similarity')}, matches={sim_result.get('matches')}",
            "evidence": sim_result.get("matches", [])
        }

        # Gatekeeper synthesizes final verdict
        final = mock.synthesize([analyst_result, similarity_analysis])

        # Add metadata and deterministic details for easier debugging in dev
        final["masked_user_id"] = masked_user_id
        final["input_summary"] = {"text_length": len(tweet_text or ""), "image_provided": bool(image_url)}
        final["analyst"] = analyst_result
        final["similarity"] = sim_result

        return final