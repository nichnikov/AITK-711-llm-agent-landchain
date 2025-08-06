from src.chains.processing_chain import load_prompts_from_config, create_processing_chain

prmts = load_prompts_from_config()
print(prmts)

r = create_processing_chain()
print(r)
