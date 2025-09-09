import json
import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
import pytesseract
from PIL import Image
import re
from sympy import symbols, Eq, solve

# Load JSON Database
with open("questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
solutions = [item["solution"] for item in data]

# Load Bangla Sentence Transformer
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(questions, convert_to_tensor=True)

# Bangla Digit Normalization
bangla_to_arabic = {'০':'0','১':'1','২':'2','৩':'3','৪':'4','৫':'5','৬':'6','৭':'7','৮':'8','৯':'9'}
def normalize_digits(text):
    return ''.join([bangla_to_arabic.get(c, c) for c in text])

# OCR Function
def extract_text(image):
    text = pytesseract.image_to_string(image, lang='ben')
    text = re.sub(r'\n+', ' ', text).strip()
    text = normalize_digits(text)
    return text

# Equation Solver
def solve_equation(text):
    text = normalize_digits(text)
    var_match = re.findall(r'[a-zA-Z]', text)
    if not var_match:
        return None
    var = symbols(var_match[0])
    try:
        lhs_rhs = text.split('=')
        lhs = lhs_rhs[0]
        rhs = lhs_rhs[1]
        eq = Eq(eval(lhs), eval(rhs))
        sol = solve(eq, var)
        steps = [
            f"ধাপ ১: সমীকরণ লেখা হলো {lhs} = {rhs}",
            f"ধাপ ২: SymPy সমাধান ব্যবহার করে হিসাব করা হলো",
            f"ধাপ ৩: {var} = {sol[0]}"
        ]
        return "\n".join(steps)
    except:
        return None

# Retrieval Function
def get_answer(input_text):
    input_emb = model.encode([input_text], convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(input_emb, embeddings)
    idx = torch.argmax(cos_scores).item()
    answer = solutions[idx]
    if isinstance(answer, list):
        return "\n".join([f"ধাপ {i+1}: {step}" for i, step in enumerate(answer)])
    else:
        steps = re.split(r'[।.!?]', answer)
        steps = [s.strip() for s in steps if s.strip()]
        return "\n".join([f"ধাপ {i+1}: {s}" for i, s in enumerate(steps)])

# Combined Function for Gradio
def solve_question(text_input, image_input):
    if image_input is not None:
        text_input = extract_text(image_input)
    if not text_input:
        return "অনুগ্রহ করে প্রশ্ন লিখুন বা ছবি আপলোড করুন।"
    
    eq_solution = solve_equation(text_input)
    if eq_solution:
        return eq_solution
    
    return get_answer(text_input)

# Gradio Interface
iface = gr.Interface(
    fn=solve_question,
    inputs=[
        gr.Textbox(label="Bangla Question", placeholder="এখানে প্রশ্ন লিখুন..."),
        gr.Image(label="Upload Question Image", type="pil", optional=True)
    ],
    outputs=gr.Textbox(label="Step-by-Step Solution"),
    title="Bangla AI Question Solver",
    description="Type a Bangla STEM question or upload a picture. It can solve equations with variables or retrieve solutions step by step."
)

if __name__ == "__main__":
    iface.launch()
