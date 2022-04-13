from transformers import TFGPT2LMHeadModel, GPT2Tokenizer


if __name__ == '__main__':

	'''import pre-trained model'''
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
	model = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)

	'''encode user input'''
	sentence = input("Type your sentence : ")
	input_sentence = tokenizer.encode(sentence, return_tensors='tf')

	'''generate sentences based on top-k and top-p statistical models'''
	sample = model.generate(input_sentence,
							do_sample=True,
							max_length=300,
							temperature=0.7,
							top_k=50,
							top_p=0.95,
							num_return_sequences=3
							)

	for i, e in enumerate(sample):
		print(f'{i}. {tokenizer.decode(e, skip_special_tokens = True)}')