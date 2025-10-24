import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# --- Environment Variable Check ---
serper_api_key = os.getenv("SERPER_API_KEY")
if not serper_api_key:
    raise EnvironmentError("SERPER_API_KEY not found. Please set it in your .env file or Streamlit secrets.")

search_tool = SerperDevTool()

# --- LLM Configuration ---
# Use crewai's native LLM class for Google
gemini_llm = LLM(
    model='gemini/gemini-2.5-flash',
    temperature=0.1
)

# --- Agent Definitions (FINETUNED) ---

# 1. YouTube Guideline Expert
guideline_expert = Agent(
    role='YouTube Cryptocurrency Content Policy Expert',
    goal="""Search for the most current YouTube community guidelines for 'Spam, Deceptive 
    Practices & Scams' and 'Harmful or Dangerous Content'. 
    **Crucially, also search for the *nuance* of how these policies are 
    applied to crypto content, distinguishing between 'market analysis' 
    and 'financial advice'**.""",
    backstory="""You are a compliance expert who understands YouTube's policies 
    and the crypto community. You know creators aren't trying to scam, but 
    might accidentally trigger policies. Your goal is to find the *exact line* between passionate market analysis and a policy violation.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=gemini_llm
)

# 2. Crypto Content Analyst (FINETUNED)
content_analyst = Agent(
    role='Cryptocurrency Video Content Analyst',
    goal="""Analyze a video transcription to identify *potential* policy risks. 
    **Your primary job is to distinguish between standard crypto market 
    commentary (like on-chain analysis, 'whale watching', price speculation) 
    and genuine high-risk claims ('guaranteed profit', 'buy this now', 
    'insider info').** Rate each potential issue as Low, Medium, or High risk.""",
    backstory="""You are a seasoned crypto content analyst. You understand the jargon. 
    You don't just flag keywords; you analyze *intent*. You know 'whale 
    watching' is analysis, but presenting it as 'secret insider knowledge' 
    is a risk. Your goal is to help the creator, not to 'bust' them. 
    You must be pragmatic.""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# 3. Report Writer (FINETUNED)
report_writer = Agent(
    role='Constructive Compliance Report Writer',
    goal="""Generate a concise, actionable report in Markdown. The report must 
    summarize risks and provide *constructive, safe alternative phrasing*. 
    The tone must be helpful and collaborative, not purely critical. 
    The final 'Risk Score' should reflect a realistic, not an 
    over-exaggerated, assessment.""",
    backstory="""You are a helpful advisor. You synthesize the policy nuances and 
    the content analysis into a practical guide. Your 'suggestions' 
    are the most important part. You offer *safer ways to say the same thing* (e.g., "Instead of 'this is a good buy-in opportunity,' try 
    'this is a level I'm personally watching.'").""",
    verbose=True,
    allow_delegation=False,
    llm=gemini_llm
)

# --- Task Definitions (FINETUNED) ---
def create_crew(video_transcript, video_language):
    
    # Task 1: Research current policies
    research_task = Task(
        description=f"""Search for the most up-to-date YouTube community guidelines 
        regarding cryptocurrency. Focus on the *nuance* of how 'financial advice', 
        'scams', and 'harmful content' policies are applied. What is the 
        difference between analysis and advice? The video language is 
        {video_language}.""",
        expected_output="""A bulleted list of key policy red flags, with a focus 
        on the subtle differences between allowed analysis and banned advice.""",
        agent=guideline_expert
    )

    # Task 2: Analyze the provided transcript
    analyze_task = Task(
        description=f"""Analyze the following video transcript:
        ---
        TRANSCRIPT:
        {video_transcript}
        ---
        Cross-reference this text against the policy red flags. For each 
        potential issue, provide a 'Risk Rating' (Low, Medium, High) and a 
        brief justification.
        
        **IMPORTANT: Do NOT flag standard crypto analysis (like 'on-chain data', 
        'whale wallet moved', 'I'm bullish on...') as high risk unless it's 
        combined with a guarantee or a direct call to buy.** Differentiate between normal crypto creator enthusiasm and 
        deceptive 'get rich quick' promises.""",
        expected_output="""A list of potentially problematic phrases, each with a 
        realistic Risk Rating (Low, Medium, High) and a justification.""",
        agent=content_analyst
    )

    # Task 3: Write the final report
    report_task = Task(
        description="""Compile all findings into a final, constructive compliance report. 
        The report must be in Markdown format and include:
        1.  A brief summary of the *key* policy risks found.
        2.  A list of *specific quotes* from the transcript that are risky, 
            along with their Risk Rating.
        3.  A final 'Overall Risk Score' (Clear, Low, Medium, High).
        4.  **Actionable, bullet-pointed suggestions for how to *rephrase* or 
            *add context* to the risky quotes to reduce the violation risk 
            *while maintaining the video's core message*.**
        """,
        expected_output="""A final, polished compliance report in Markdown, 
        with a focus on helpful, alternative phrasing.""",
        agent=report_writer
    )

    # --- Crew Definition ---
    return Crew(
        agents=[guideline_expert, content_analyst, report_writer],
        tasks=[research_task, analyze_task, report_task],
        process=Process.sequential,
        verbose=True
    )

