import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer, Pipeline
from fastapi import FastAPI

app = FastAPI()

model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")

tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)


sentence = """
Việc cậu bé nhảy lầu tự tử và để lại bức thư cho cha mẹ. Những bậc cha mẹ đó thay vì được nhận sự an ủi và đồng cảm thì lại bị đem ra thành vấn đề chỉ trích: chắc hẳn là ông bố bà mẹ này đã tạo nên áp lực rất nhiều cho đứa trẻ, làm cha làm mẹ mà không biết dạy con.
"""

# Just like PhoBERT: INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
input_ids = torch.tensor([tokenizer.encode(sentence)])

# pipeline = Pipeline(model=model, tokenizer=tokenizer, input_ids=input_ids)

@app.post('/predict')
async def predict(sentence: str):
    out = model(input_ids)
    prediction = out.logits.softmax(dim=-1).tolist()
    max_prob = max(prediction)
    if max_prob == prediction[0]:
        return 'Negative'
    elif max_prob == prediction[1]:
        return 'Positive'
    return 'Neutral'



# with torch.no_grad():
#     out = model(input_ids)
#     results = out.logits.softmax(dim=-1).tolist()
    
    # Output:
    # [[0.002, 0.988, 0.01]]
    #     ^      ^      ^
    #    NEG    POS    NEU
