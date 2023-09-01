from FACT_GEN import fact_gen

with open("keys.txt", "r") as f:
    serper, openai = f.read().splitlines()

    
output_path = "test_output.txt"
pipeline = fact_gen(openai, serper, output_file=output_path)

name = "Rob Furlong"
prompt = f"Write a bio of {name}."

res_str = pipeline.generate(prompt)
with open(output_path, "a") as f:
    print(res_str, file=f)