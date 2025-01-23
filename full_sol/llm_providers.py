from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline

def build_hf_llm(model_name: str = "tiiuae/falcon-7b-instruct"):
    """
    Build a HuggingFace LLM pipeline with a more advanced instruct model.
    Adjust model_name to your chosen model. 
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True
    )
    
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return llm
