import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

# Pydantic for schema definition
from pydantic import BaseModel, Field

# LangChain components for building the generation chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

# --- Load environment variables for API keys ---
load_dotenv()

# =====================================================================================
# SECTION 1: SCHEMA DEFINITION (Unchanged)
# =====================================================================================
class SentimentPreference(BaseModel):
    """Defines the strict schema for the model's JSON output."""
    chosen: str = Field(..., description="A high-quality, accurate, and nuanced analysis of the review's sentiment.")
    rejected: str = Field(..., description="A plausible but inferior analysis that is inaccurate, simplistic, or misses context.")

# =====================================================================================
# SECTION 2: DATASET GENERATION LOGIC (Refactored with LangChain)
# =====================================================================================
def generate_preference_dataset(output_file: str, num_samples: int = 1000) -> None:
    """
    Generates a preference dataset using a robust LangChain chain and saves it to a JSON file.
    """
    # 1. --- SETUP LANGCHAIN COMPONENTS ---
    # The language model we'll use
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.5)

    # The parser responsible for validating output against our Pydantic schema
    pydantic_parser = PydanticOutputParser(pydantic_object=SentimentPreference)

    # A robust parser that can auto-correct malformed output from the LLM
    output_fixer = OutputFixingParser.from_llm(llm=llm, parser=pydantic_parser)

    # The prompt template, which automatically includes formatting instructions from the parser
    prompt_template = PromptTemplate(
            template="""You are an expert data labeler. Your task is to generate a high-quality preference pair for training a reward model.
    The 'chosen' response must be a comprehensive and nuanced summary of the review's sentiment.
    The 'rejected' response must be a plausible but clearly inferior summary that contains a specific flaw.

    Here are the types of flaws to introduce into the 'rejected' response:
    - It is too simplistic or misses the main point of the review.
    - It is factually inaccurate and misrepresents details from the review.
    - It fails to capture mixed sentiment, labeling a nuanced review as purely positive or negative.
    - It focuses on a minor, unimportant detail of the review.

    ---
    EXAMPLE:
    Review: "This camera takes great photos, but the user interface is very confusing."
    Output:
    {{
        "chosen": "This is a mixed sentiment review. The reviewer praised the photo quality but criticized the confusing user interface.",
        "rejected": "The review is positive because it mentions great photos."
    }}
    ---
    ACTUAL TASK:
    Review:
    {review}

    {format_instructions}
    """,
            input_variables=["review"],
            partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
        )

    # The complete chain that pipes all components together
    # This defines the flow: format prompt -> call LLM -> parse/fix output
    chain = prompt_template | llm | output_fixer

    # 2. --- DATA LOADING & PROCESSING ---
    print("Loading the amazon_polarity dataset...")
    dataset = load_dataset("amazon_polarity", split="train").shuffle(seed=42).select(range(num_samples * 2))
    
    preference_data = []
    
    print(f"Generating {num_samples} preference pairs using the LangChain chain...")
    for example in tqdm(dataset, total=num_samples):
        if len(preference_data) >= num_samples:
            break
            
        review_content = example['content']
        
        try:
            # 3. --- INVOKE THE CHAIN ---
            # This single call handles prompt formatting, API interaction, parsing, and auto-correction.
            response_model = chain.invoke({"review": review_content})
            
            # Append the validated data as a standard dictionary
            entry = {
                "review": review_content,
                **response_model.model_dump() # Converts the Pydantic model to a dict
            }
            preference_data.append(entry)
            
        except Exception as e:
            # This will only trigger if the OutputFixingParser also fails
            print(f"\nSkipping sample. Chain failed even after attempting to fix: {e}")
            continue

    # 4. --- SAVING THE DATA ---
    print(f"\nSaving {len(preference_data)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, indent=4)
        
    print("Dataset generation complete.")
    print("IMPORTANT: Remember to manually review and clean the generated data for quality.")

# =====================================================================================
# SECTION 3: SCRIPT EXECUTION
# =====================================================================================
if __name__ == "__main__":
    generate_preference_dataset("RLHF_data_for_sentiment_product_review.json", num_samples=500)