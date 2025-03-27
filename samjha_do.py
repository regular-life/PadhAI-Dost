from langchain_google_genai import GoogleGenerativeAI


def samjha_do(text, prior_knowledge, api_key):
    """Explains the document in detail based on prior knowledge."""
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

    if prior_knowledge == "Beginner":
        prompt = f"""Create a complete educational summary of this text for someone completely new to the subject:
        {text}
        
        - Start with a 1-sentence overview that defines the core concept in everyday language
        - Break down complex ideas using step-by-step explanations and relatable analogies
        - Explicitly define all technical terms using simple language (format: *Term*: Definition)
        - Include concrete examples for abstract concepts
        - Highlight connections between ideas using arrows (→) to show relationships
        - End with "Key Takeaways" bullet points summarizing fundamental principles
        - Add "Common Questions" section anticipating beginner misunderstandings"""

    elif prior_knowledge == "Intermediate":
        prompt = f"""Generate a structured knowledge enhancement summary of this text:
        {text}
        
        - Begin with a conceptual framework diagram (described in text form)
        - Organize content using these sections: Core Principles, Current Applications, Ongoing Debates
        - Use domain-specific terminology but provide brief context reminders in parentheses
        - Include compare/contrast sections showing relationships between concepts (↔ vs. → relationships)
        - Add "Deep Dive" callout boxes explaining 2-3 non-intuitive aspects
        - Incorporate relevant historical context for key theories/methods
        - Conclude with "Connections to Foundational Knowledge" linking to basic concepts"""

    else:  # Advanced
        prompt = f"""Produce an expert-level synthesis and critical analysis of this text:
        {text}
        
        - Open with current research status and knowledge gaps in the field
        - Structure using: Theoretical Foundations, Methodological Approaches, Emerging Frontiers
        - Employ discipline-specific jargon with expectation of fluency
        - Include critical evaluation of strengths/weaknesses of major theories
        - Add "Research Implications" section forecasting future directions
        - Create cross-connections to related advanced concepts (↔ denotes bidirectional relationships)
        - Incorporate citations to seminal works (format: [Author et al., Year])
        - Highlight unresolved challenges in bullet-point form"""

    response = llm.generate([prompt])
    return response.generations[0][0].text
