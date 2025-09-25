from sentence_transformers import SentenceTransformer, util
import torch

faq_data = [
    {"question": "How to reset my password?", "answer": "Go to settings > reset password and follow instructions."},
    {"question": "What is your refund policy?", "answer": "Refunds are processed within 5 business days."},
    {"question": "How to contact support?", "answer": "You can contact support at support@example.com."},
    {"question": "What are your working hours?", "answer": "We are available Monday to Friday, 9 AM to 6 PM."}
]

# Load Sentence Transformer and Precompute FAQ Embeddings

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Context-Aware Chat Function

conversation_history = {}

def chat(user_id, user_message, top_k=1):
    """
    user_id: unique identifier for user
    user_message: latest user message
    top_k: number of FAQ results to consider
    """
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append(f"User: {user_message}")
    
    query_embedding = model.encode(user_message, convert_to_tensor=True)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, faq_embeddings) # between the existing question and user enquery
    top_results = torch.topk(cos_scores, k=top_k)
    
    best_idx = top_results[1][0].item()
    answer = faq_answers[best_idx]
    
    conversation_history[user_id].append(f"Bot: {answer}")
    
    recent_context = "\n".join(conversation_history[user_id][-6:])  # last 6 answers by bot store 
    return answer, recent_context

# run CHatbot

user_id = "user_001"  

print("Welcome to FAQ Chatbot! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    
    answer, context = chat(user_id, user_input)
    print(f"Bot: {answer}")
    
    # print("Context:\n", context)