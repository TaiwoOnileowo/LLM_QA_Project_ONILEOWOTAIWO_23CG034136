from flask import Flask, render_template, request
from LLM_QA_CLI import preprocess_input, query_llm

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    question = None
    processed_text = None
    tokens = None
    answer = None
    
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            # Reuse logic from CLI
            processed_text, tokens = preprocess_input(question)
            answer = query_llm(question)
            
    return render_template('index.html', 
                           question=question, 
                           processed_text=processed_text, 
                           tokens=tokens, 
                           answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
