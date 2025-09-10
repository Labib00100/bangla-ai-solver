import gradio as gr
import pytesseract
from PIL import Image

import re
from sympy import symbols, Eq, solve, sympify, sin, cos, tan, sqrt

# Load your Bangla Q&A database
with open("questions.json", "r", encoding="utf-8") as f:
    db = json.load(f)

# --- Detect if input is an equation ---
def is_equation(text):
    return "=" in text or re.search(r"[xyzθαβ]", text)

# --- Solve equations using Sympy ---
def solve_equation(text):
    try:
        # Replace Bangla/Greek letters with sympy-friendly versions
        text = text.replace("θ", "theta").replace("α", "alpha").replace("β", "beta")

        # Define possible symbols
        x, y, z, theta, alpha, beta = symbols('x y z theta alpha beta')

        # If equation has "=", split into left and right
        if "=" in text:
            left, right = text.split("=")
            left_expr = sympify(left)
            right_expr = sympify(right)
            eq = Eq(left_expr, right_expr)
        else:
            # If no "=", treat as expression = 0
            eq = Eq(sympify(text), 0)

        sol = solve(eq, dict=True)
        return f"সমাধান: {sol}"
    except Exception as e:
        return f"সমীকরণ সমাধান করতে সমস্যা হয়েছে ⚠️: {str(e)}"

# --- Search in JSON database ---
def search_db(query):
    for item in db:
        if item["question"] in query:
            return "\n".join(item["solution"])
    return None

# --- Main solver ---
def solve_problem(input_text, image):
    # If image uploaded → OCR first
    if image is not None:
        ocr_text = pytesseract.image_to_string(Image.open(image), lang="eng+ben")
        input_text = ocr_text.strip()

    if not input_text:
        return "কোনো প্রশ্ন দেয়া হয়নি।"

    # If looks like an equation → solve with Sympy
    if is_equation(input_text):
        return solve_equation(input_text)

    # Else → search in database
    ans = search_db(input_text)
    if ans:
        return ans

    return "দুঃখিত 😔, এই প্রশ্নটি ডাটাবেজে নেই।"

# --- Gradio interface ---
demo = gr.Interface(
    fn=solve_problem,
    inputs=[gr.Textbox(label="প্রশ্ন লিখুন"), gr.Image(type="filepath", label="অথবা ছবি আপলোড করুন")],
    outputs="text",
    title="📘 বাংলা প্রশ্ন সমাধানকারী",
    description="টেক্সট বা ছবি থেকে প্রশ্ন লিখুন, এবং সমাধান পান।"
)

if __name__ == "__main__":
    demo.launch()
