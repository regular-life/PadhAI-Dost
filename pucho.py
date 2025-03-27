from langchain_google_genai import GoogleGenerativeAI


def pucho(text, question_type, num_questions, difficulty, api_key):
    """Generates practice questions based on the document."""
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

    prompt = f"""Generate {num_questions} {question_type} questions based on the text below. Follow these guidelines:

    Text: {text}

    Requirements:
    1. Difficulty Parameters (Scale 1-10):
    - Complexity Tier:
        - 1-3 (Basic): Focus on direct information recall
        - 4-6 (Intermediate): Require analysis/application
        - 7-10 (Advanced): Demand synthesis/evaluation
        Higher the difficulty, more abstract and thought-provoking are the questions

    2. Question Architecture:
    {"- For Objective:" if question_type == "Objective" else "- For Subjective:"}
        {f'''{{
        "Objective": [
            "Use unambiguous question phrasing",
            "Avoid trick questions for levels <5",
            "Include 15% graph/data interpretation questions if applicable",
            "Balance factual vs conceptual (60/40 ratio)"
        ],
        "Subjective": [
            "Require evidence-based reasoning",
            "Include 25% scenario-based prompts",
            "Add comparative analysis elements for levels >7",
            "Incorporate ethical considerations for levels ≥8"
        ]
        }}[{question_type}]'''}

    3. Cognitive Demand Matrix:
    - Use Bloom's verbs according to difficulty:
        1-3: Remember/Understand
        4-6: Apply/Analyze
        7-10: Evaluate/Create

    4. Formatting Rules:
    - Never include answers or scoring guidelines
    - For Objective:
        * Multiple Choice: 4 plausible options without answer markers
        * True/False: No explanations
        * Matching: Maximum 6 item pairs
    - For Subjective:
        * Case Studies: Include 3 contextual variables
        * Essays: Suggest 2-3 required citation points
        * Problem-Solving: Provide necessary data sets

    5. Quality Checks:
    - Ensure 1:2:1 ratio of concrete:applied:abstract questions
    - Include 2 verification questions for consistency
    - Avoid cultural/geographic bias in framing

    6. Additional Instructions:
    - Use inclusive language for all questions
    - Maintain a neutral tone throughout
    - Do not repeat questions with similar content
    - Ensure questions are relevant to the text
    - Avoid ambiguous phrasing or vague pronouns
    - Keep questions concise and to the point
    - Incorporate real-world examples where possible
    - Use proper grammar and punctuation
    - Verify all questions for factual accuracy
    - Provide clear instructions for each question
    - Make sure to only mention the questions, not the answers (Also, no need to show difficulty level, etc.)
    """

    response = llm.generate([prompt])
    questions = response.generations[0][0].text.split(
        "\n")  # Simple split, may need refinement
    return questions
