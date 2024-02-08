REWRITE = """Rewrite the following ariticle, maintaining its original meaning. Do not add any new information. Keep the specified words unchanged.
Words to Keep Unchanged: {ANSWERS}
Original Article: {CONTEXT}
Rewritten Article:"""

sentence_format = """Please write a single sentence claim using the follwoing question and answer. The claim should include the answer and be as realistic as possible:
Question: {QUESTION}
Answer: {ANSWER}
Claim:"""
claim_format = """Given a claim, please write a concise, factual passage to support it. You can make up fake content and supporting evidence but it should be as realistic as possible. The passage must be less than 100 words.
Claim: {CLAIM}
Passage:"""