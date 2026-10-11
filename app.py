from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

#initialize our fastapi app
app = FastAPI(title="Text Summarizer",description="Text Summarization using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("device: ",device)
model.to(device)

#templating
templates = Jinja2Templates(directory=".")

#Input schema for dialogue => string

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_dialogue(dialogue : str)->str:
    model.to(device)
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,padding='max_length',
        max_length = 512,
        truncation = True,
        return_tensors = 'pt'
    )

    inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}
    
    
    # generate the summary => token ids
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True
    )

    # token ids convert to text => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)