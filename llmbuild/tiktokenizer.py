import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Akwirw ier"
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
for item in integers:
    print(tokenizer.decode([item]))
strings = tokenizer.decode(integers)
print(strings)
