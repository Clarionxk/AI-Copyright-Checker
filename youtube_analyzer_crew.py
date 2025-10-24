import os
# --- THIS IS THE FIX (Line 1) ---
# We import LLM from crewai instead of ChatGoogleGenerativeAI from langchain
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# --- Environment Variable Check ---
# Check for SERPER_API_KEY for the search tool
serper_api_key = os.getenv("SERPER_API_KEY")
if not serper_api_key:
    raise EnvironmentError("SERPER_API_KEY not found. Please set it in your .env file or Streamlit secrets.")

# Initialize the search tool
search_tool = SerperDevTool()

# --- LLM Configuration ---
# --- THIS IS THE FIX (Line 2) ---
# We use crewai's LLM class and specify the provider 'gemini/'
gemini_llm = LLM(
    model='gemini/gemini-2.5-flash',
    temperature=0.1
)
# --- END OF FIX ---


# --- Agent Definitions ---

# 1. YouTube Guideline Expert
guideline_expert = Agent(
    role='YouTube Cryptocurrency Content Policy Expert',
    goal="""Search Google for the most current YouTube community guidelines 
    and policies, specifically focusing on 'Spam, Deceptive Practices & Scams', 
    'Harmful or Dangerous Content', and any rules related to finance, 
    cryptocurrency, and NFTs.""",
    backstory="""You are a compliance officer who is an expert on YouTube's 
    Community Guidelines. You have deep knowledge of how these rules are 
    applied to cryptocurrency channels. You MUST use your search tool to find 
    the *latest* policy information.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=gemini_llm
)

# 2. Crypto Content Analyst
content_analyst = Agent(
    role='Cryptocurrency Video Content Analyst',
    goal="""Analyze a given video transcription to identify any specific phrases, 
    claims, or calls-to-action that could be flagged by YouTube's policies. 
    You must look for 'get rich quick' schemes, exaggerated promises, 
    undisclosed sponsorships, and any form of financial advice.""",
    backstory="""You are a meticulous analyst who specializes in crypto content. 
    You read video transcripts and cross-reference them with YouTube's policies 
    to find potential violations. You are looking for specific, actionable 
    examples from the text.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# 3. Report Writer
report_writer = Agent(
    role='Compliance Report Writer',
    goal="""Generate a concise, actionable compliance report in Markdown format. 
    The report must summarize the policy findings from the Guideline Expert and 
    the specific content risks from the Content Analyst. It must include a 
    final 'Risk Score' (Clear, Low, Medium, High) and provide clear, 
    bullet-pointed suggestions for how to fix any potential violations.""",
    backstory="""You are a clear and concise writer who creates reports for 
    content creators. Your reports are easy to understand and provide 
    practical advice. You synthesize information from both the policy expert 
    and the content analyst into a single, final report.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# --- Task Definitions ---
# The function `create_crew` now accepts the transcript and language
def create_crew(video_transcript, video_language):
    
    # Task 1: Research current policies
    # This task no longer needs video_language, but we'll leave it in the
    # description in case it's useful context for the agent.
    research_task = Task(
        description=f"""Search for the most up-to-date YouTube community guidelines 
        regarding cryptocurrency, scams, and financial advice. Also, search for any 
        recent news or blog posts about YouTube cracking down on crypto channels. 
        The video language is {video_language}, which may be relevant for 
        region-specific policies, but the primary search should be in English.
        """,
        expected_output="A bulleted list of key policy points and red flags to look for.",
        agent=guideline_expert
    )

    # Task 2: Analyze the provided transcript
    analyze_task = Task(
        description=f"""Analyze the following video transcript:
        ---
        TRANSCRIPT:
        {video_transcript}
        ---
        Cross-reference this text against the policy red flags. Identify every 
        specific phrase or claim that could be a violation. Pay special attention 
        to promises of returns, financial advice, or phrases that sound like 
        'get rich quick' schemes.""",
        expected_output="A list of potentially problematic phrases and an explanation of why they are risky.",
        agent=content_analyst
    )

    # Task 3: Write the final report
    report_task = Task(
        description="""Compile all findings into a final compliance report. 
        The report must be in Markdown format and include:
        1.  A brief summary of the *current* YouTube crypto policies.
        2.  A list of *specific quotes* from the transcript that are high-risk.
        3.  A final 'Risk Score' (Clear, Low, Medium, High).
        4.  Actionable, bullet-pointed suggestions for how to fix the issues.
        """,
        expected_output="A final, polished compliance report in Markdown.",
        agent=report_writer
    )

    # --- Crew Definition ---
    return Crew(
        agents=[guideline_expert, content_analyst, report_writer],
        tasks=[research_task, analyze_task, report_task],
        process=Process.sequential,
        # --- THIS IS THE FIX ---
        verbose=True  # Changed from 2 to True
        # --- END OF FIX ---
    )

