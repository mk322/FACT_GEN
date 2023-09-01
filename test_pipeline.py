from FACT_GEN import fact_gen


output_path = "test_output.txt"
pipeline = fact_gen(output_file=output_path)

name = "Rob Furlong"
prompt = f"Write a bio of {name}."

res_str = pipeline.generate(prompt)
with open(output_path, "a") as f:
    print(res_str, file=f)