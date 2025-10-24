import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# --- IMPORTANT ---
# This file now uses the Google Gemini API.
# Make sure you have environment variables for BOTH keys set.
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
# os.environ["SERPER_API_KEY"] = "YOUR_API_KEY_HERE"

# Check for the API keys
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
if not os.getenv("SERPER_API_KEY"):
    raise EnvironmentError("SERPER_API_KEY environment variable not set. Get one from serper.dev")

# --- LLM DEFINITION ---
# Initialize the Gemini LLM
# We'll use gemini-2.5-flash-preview-09-2025 as it's fast and capable.
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-09-2025",
    temperature=0.2,
    # Make sure to pass your API key if it's not set as a default environment variable
    # google_api_key=os.environ["GOOGLE_API_KEY"] 
)

# Initialize the search tool
search_tool = SerperDevTool()

# --- AGENT DEFINITIONS ---

# 1. The Guideline Expert Agent
# This agent is your specialist in YouTube's policies, especially for crypto.
guideline_expert = Agent(
  role='YouTube Policy Expert for Cryptocurrency',
  goal="""Provide the most critical YouTube community guidelines
  that crypto channels must follow. Focus on spam, deceptive practices,
  scams, and harmful financial content.""",
  backstory="""You are a world-class policy analyst who has memorized
  all of YouTube's community guidelines. You have a special focus
  on how these rules are applied to cryptocurrency and financial content.
  You know all the red flags, from "exaggerated promises" and "get rich quick"
  language to "cryptophishing" and linking to unregulated exchanges.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool],
  llm=gemini_llm  # <-- Pass the Gemini LLM to the agent
)

# 2. The Video Compliance Agent
# This agent analyzes a specific video transcript against the guidelines.
video_analyzer = Agent(
  role='Video Compliance Analyzer',
  goal="""Analyze a given video transcript against a set of YouTube policies.
  Identify specific lines or phrases that could be flagged as violations.
  Provide a detailed report with the problematic text, the policy it
  might violate, and a risk level (Low, Medium, High).""",
  backstory="""You are a meticulous compliance officer. You take a video
  transcript and a list of rules, and you cross-reference them with extreme
  attention to detail. You don't make assumptions; you quote the evidence
  directly from the transcript and cite the specific rule that might
  be at risk.""",
  verbose=True,
  allow_delegation=False,
  llm=gemini_llm  # <-- Pass the Gemini LLM to the agent
)

# --- TASK DEFINITIONS ---

def create_crew(video_transcript, video_language):
    """
    Creates and configures the CrewAI crew to analyze the video.

    Args:
        video_transcript (str): The text transcript of the video.
        video_language (str): The language of the transcript.

    Returns:
        Crew: The configured CrewAI crew.
    """

    # Task 1: Research the relevant policies
    # This task is dynamic based on the language.
    research_task = Task(
      description=f"""
      1. Find the most up-to-date YouTube community guidelines related
         to cryptocurrency, financial advice, scams, and deceptive practices.
      2. Pay special attention to policies on "cryptophishing," "exaggerated promises,"
         and "get rich quick" schemes.
      3. Summarize these rules into a concise, actionable checklist.
      4. Conduct your search and formulate the checklist in {video_language}.
      """,
      expected_output="""A concise checklist of the top 5-7 YouTube policy
      red flags for a crypto video, written in {video_language}.""",
      agent=guideline_expert
    )

    # Task 2: Analyze the transcript
    analysis_task = Task(
      description=f"""
      Using the policy checklist from the expert, analyze the following
      video transcript (which is in {video_language}).

      Scan the entire transcript for any text that matches the red flags.
      This includes:
      - Any "guarantees" of profit or "get rich quick" promises.
      - Any requests for users to send cryptocurrency or share wallet details (cryptophishing).
      - Any links to unverified exchanges or platforms without a proper disclaimer.
      - Any hype or exaggerated claims about a coin or project without
        clear disclaimers that this is not financial advice.

      Transcript to Analyze:
      ---
      {video_transcript}
      ---
      """,
      expected_output="""A detailed compliance report in {video_language}.
      The report must be in Markdown format and include:
      1.  **Overall Risk Assessment:** (Low, Medium, or High)
      2.  **Potential Issues Found:**
          - **Timestamp/Quote:** (Quote the exact problematic text. If timestamps aren't available, just use the quote.)
          - **Potential Policy Violation:** (e.g., "Spam and Deceptive Practices: Exaggerated Promises")
          - **Risk:** (Low/Medium/High)
          - **Suggestion:** (e.g., "Add a clear 'This is not financial advice' disclaimer here.")
      3.  **Final Summary:** A brief conclusion of the findings.

      If no issues are found, state "No significant policy risks detected."
      """,
      agent=video_analyzer,
      context=[research_task] # This task depends on the output of the research_task
    )

    # --- CREW DEFINITION ---
    youtube_crew = Crew(
      agents=[guideline_expert, video_analyzer],
      tasks=[research_task, analysis_task],
      process=Process.sequential,
      verbose=2
      # Note: You can also pass the 'llm' here to the Crew,
      # and it will be used by all agents that don't have one specified.
      # llm=gemini_llm 
    )

    return youtube_crew

if __name__ == "__main__":
    # This is a test run. The actual transcript will come from the Streamlit app.
    print("--- Starting CrewAI Test Run ---")
    
    # Mock transcript in Spanish
    TEST_LANGUAGE = "Spanish"
    TEST_TRANSCRIPT = """
    ¡Hola a todos! Hoy, les voyG a mostrar una gema cripto que
    literalmente te hará rico mañana. Es una garantía.
    Esta moneda va a la luna, 1000x seguro.
    Rápido, envía 1 ETH a esta billetera en la descripción
    para entrar en la preventa exclusiva. ¡No te lo pierdas!
    ¡Haz clic en el enlace de abajo ya!
    """

    crew = create_crew(TEST_TRANSCRIPT, TEST_LANGUAGE)
    result = crew.kickoff()

    print("\n\n--- CrewAI Test Run Finished ---")
    print("\n--- Final Report ---")
    print(result)


