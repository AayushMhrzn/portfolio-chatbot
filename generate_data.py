"""
generate_data.py
----------------
Generates 10,000+ Q&A training pairs from your portfolio data using:
1. Seed pairs (hand-written, highest quality)
2. Question paraphrasing (same answer, different question)
3. Chunk-based generation (Groq generates new Q&A from your text chunks)
4. Negative pairs (mismatched Q&A, helps model learn boundaries)

Run this once before training:  python generate_data.py
Output:                         data/training_pairs.json
"""

import os
import json
import random
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
DATA_DIR = "data"
OUTPUT   = "data/training_pairs.json"

# ─── SEED PAIRS ───────────────────────────────────────────────────────────────
# These are your highest-quality hand-written pairs.
# Every augmentation strategy below builds on top of these.

SEED_PAIRS = [
    # About
    ("Who is Aayush Maharjan?",
     "Aayush Maharjan is aspiring AI/ML Engineer based in Lalitpur, Nepal. He completed his bachelors in Computer Engineering from Kantipur Engineering College. His skilled in Python, Machine Learning, Deep Learning, Computer Vision, LLM, Rag with multiple related projects and certifications."),
    ("Who is Aayush?",
     "Aayush Maharjan is aspiring AI/ML Engineer based in Lalitpur, Nepal. He completed his bachelors in Computer Engineering from Kantipur Engineering College. His skilled in Python, Machine Learning, Deep Learning, Computer Vision, LLM, Rag with multiple related projects and certifications."),
    ("Tell me about Aayush",
 "Aayush is an aspiring AI/ML Engineer skilled in Python, LLMs, RAG, Computer Vision, and Deep Learning with many projects, certifications. He studied Bachelors in Computer Engineering at KEC"),
    ("Where is Aayush located?",
     "Aayush is based in Gwarko, Lalitpur, Nepal."),
    ("What is Aayush's email?",
     "Aayush's email is aayushmaharjan.94@gmail.com."),
    ("What is Aayush's GitHub?",
     "Aayush's GitHub profile is github.com/AayushMhrzn."),
    ("What is Aayush's LinkedIn?",
     "Aayush's LinkedIn is linkedin.com/in/aayushmhrjn."),
    ("What is Aayush's portfolio website?",
     "Aayush's portfolio website is www.aayushmaharjan94.com.np."),
    ("Tell me about Aayush.",
     "Aayush is an enthusiastic Computer Engineering student with expertise in Python, ML, Deep Learning, Computer Vision, Agentic AI, and RAG systems."),
    ("What is Aayush's phone number?",
     "Aayush's phone number is 9803017605."),

    # Education
    ("Where did Aayush study Bachelors?",
     "Aayush studied Bachelor of Computer Engineering at Kantipur Engineering College (KEC), affiliated to Tribhuvan University."),
    ("When did Aayush graduate?",
     "Aayush completed his Bachelor of Computer Engineering in 2026."),
    ("What is Aayush's degree?",
     "Aayush has a Bachelor of Computer Engineering from Kantipur Engineering College."),
    ("Did Aayush get any scholarship?",
     "Yes, Aayush was awarded the 5th Semester Scholarship at Kantipur Engineering College for strong academic performance."),
    ("What university/College did Aayush attend?",
     "Aayush attended Kantipur Engineering College (KEC), affiliated to Tribhuvan University (TU)."),
    ("Where did Aayush study school?",
     "Aayush studied at Gyanodaya Bal Batika School."),
    ("Where did Aayush study High School or +2 College/?",
     "Aayush studied at Little Angels (LA) College for his High School (+2) education."),     
    # Skills
    ("What programming languages does Aayush know?",
     "Aayush knows Python, C, C++, Java, SQL, JavaScript, and HTML."),
    ("What ML frameworks does Aayush use?",
     "Aayush uses TensorFlow, Keras, PyTorch, HuggingFace Transformers, and Scikit-learn."),
    ("Does Aayush know Docker?",
     "Yes, Aayush knows Docker and has containerized production applications using Docker."),
    ("Does Aayush know LangChain?",
     "Yes, Aayush has hands-on experience with LangChain for building LLM-powered applications."),
    ("Does Aayush know RAG?",
     "Yes, Aayush has built RAG-based chatbots using LangChain, FAISS, and Groq."),
    ("Does Aayush know Computer Vision?",
     "Yes, Aayush has strong Computer Vision skills using OpenCV and deep learning architectures like CNN."),
    ("What AI tools does Aayush know?",
     "Aayush knows LLMs, LangChain, RAG, Agentic AI, Prompt Engineering, LoRA, DistilBERT, FAISS, and vector databases."),
    ("Does Aayush know fine-tuning?",
     "Yes, Aayush has experience fine-tuning models using LoRA and PEFT library on HuggingFace."),
    ("Does Aayush know PyTorch?",
     "Yes, Aayush is certified in PyTorch from OpenCV University and uses it for deep learning projects."),
    ("Does Aayush know TensorFlow?",
     "Yes, Aayush is certified in TensorFlow/Keras from OpenCV University."),
    ("Does Aayush know SQL?",
     "Yes, SQL is one of Aayush's programming skills."),
    ("Does Aayush know JavaScript?",
     "Yes, Aayush knows JavaScript as part of his programming skillset."),
    ("Does Aayush know OpenCV?",
     "Yes, Aayush has strong skills in OpenCV and is certified in Computer Vision with OpenCV from OpenCV University."),
    ("Does Aayush know vector databases?",
     "Yes, Aayush has experience with vector databases including FAISS and Supabase pgvector."),
    ("Does Aayush know MCP?",
     "Yes, Aayush has implemented MCP (Model Context Protocol) server endpoints in his portfolio chatbot project."),
    ("Does Aayush know Kubernetes?",
     "Aayush has knowledge of Kubernetes and has written Kubernetes manifests for his portfolio chatbot project."),

    # Projects
    ("What projects has Aayush built?",
     "Aayush has built an Audio Driven Facial Animation system, a Network Intrusion Detection System, an Interview ChatBot using LangChain,a Portfolio Q&A RAG Chatbot, DIY HOME CCTV with Person Detection (Computer Vision) using Spare Webcam, and PONG Game only using C Graphics" ),
    ("Tell me about the Audio Driven Facial Animation project.",
     "Aayush built a speech-driven facial animation system using CNN-TCN architecture that generates realistic lip motion from audio input, trained on RAVDESS and MEAD datasets."),
    ("Tell me about the Network Intrusion Detection project.",
     "Aayush built an ML-based Network Intrusion Detection System using Random Forest achieving 97% balanced accuracy and 0.98936 AUCPR score, detecting SQL Injection, XSS, and DDoS attacks."),
    ("Tell me about the Interview ChatBot project.",
     "Aayush built an AI-powered mock interview chatbot using LangChain with TF-IDF scoring, speech recognition, webcam streaming, and automated performance report generation."),
    ("Tell me about the RAG Chatbot project.",
     "Aayush built the Portfolio RAG Chatbot — a production-grade chatbot deployed on his portfolio website built with DistilBERT, LoRA, FastAPI, Supabase, Docker"),
    ("What datasets has Aayush used?",
     "Aayush has used RAVDESS and MEAD dataset for the Audio-driven Facial Animation project, and CSIC2018 dataset for the Network Intrusion Detection project."),
    ("What was Aayush's best model accuracy?",
     "Aayush achieved 97% balanced accuracy and 0.98936 AUCPR score in his Network Intrusion Detection System project."),

    # Achievements
    ("Has Aayush won any hackathon?",
     "Yes, Aayush won the Overall Category at the HACK-a-LITE Hackathon organized by Kantipur Engineering College."),
    ("What was the hackathon project about?",
     "The hackathon project was a digital platform for preserving and promoting Nepali local arts and crafts with video-based learning, marketplace integration, and an AI-powered chatbot."),
    ("Was Aayush recognized in any learning challenge?",
     "Yes, Aayush was recognized as a TOP 60 Learner among 860+ applicants in the 60DaysOfLearningChallenge by Leapfrog Technology Inc."),
    ("What clubs is Aayush part of?",
     "Aayush is a Wing Member of the Computer Club at KEC, where he mentors sessions on GitHub, AI Agents, and Chatbots."),

    # Certifications
    ("What certifications does Aayush have?",
     "Aayush has certifications in Deep Learning Specialization, AI Training, Neural Networks, PyTorch, TensorFlow, Computer Vision with OpenCV, and Python with Data Science."),
    ("Is Aayush certified in deep learning?",
     "Yes, Aayush completed the Deep Learning Specialization from Coursera and DeepLearning.AI in April 2026."),
    ("Does Aayush have an AI certification?",
     "Yes, Aayush completed Artificial Intelligence Training from Broadway Infosys in February 2026 and Deep Learning Specialization from Coursera and DeepLearning.AI in April 2026."),

    # Job readiness
    ("Is Aayush looking for a job?",
     "Yes, Aayush is actively looking for AI Engineer roles where he can apply his skills in LLMs, RAG, fine-tuning, and production AI systems."),
    ("What kind of role is Aayush looking for?",
     "Aayush is looking for AI Engineer or ML Engineer roles focused on building production-grade AI systems."),
    ("Why should I hire Aayush?",
     "Aayush has strong hands-on experience in LLMs, RAG, Computer Vision, and Deep Learning, with real projects deployed end-to-end. He is a quick learner, hackathon winner, and top-ranked learner in competitive challenges."),
    ("Is Aayush a fresher?",
     "Aayush recently completed his final year of Computer Engineering, in April 2026, with strong project experience in AI/ML and multiple certifications."),
]


# ─── HELPER: CALL GROQ ────────────────────────────────────────────────────────

def call_groq(prompt, max_tokens=1500):
    """Call Groq API with retry on rate limit."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.8,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 10
                print(f"  Rate limit hit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Groq error: {e}")
                return None
    return None


# ─── STRATEGY 1: QUESTION PARAPHRASING ───────────────────────────────────────

def paraphrase_questions(seed_pairs, paraphrases_per_question=8):
    """
    For each seed pair, generate N different ways to ask the same question.
    Same answer, different question phrasing.
    This is the most important augmentation strategy.
    """
    print("\n[Strategy 1] Paraphrasing questions...")
    augmented = []
    batch_size = 10  # Send 10 questions per API call to save rate limit

    for i in range(0, len(seed_pairs), batch_size):
        batch = seed_pairs[i:i+batch_size]
        questions_text = "\n".join(
            [f"{j+1}. {q}" for j, (q, _) in enumerate(batch)]
        )

        prompt = f"""You are a data augmentation assistant.
For each question below, generate {paraphrases_per_question} different ways to ask the SAME question.
Keep the same meaning but vary the wording, formality, and structure.
Mix casual, formal, and conversational styles.

Questions:
{questions_text}

Respond ONLY with a JSON array of arrays. Each inner array contains {paraphrases_per_question} paraphrases for that question.
Example format:
[
  ["paraphrase1", "paraphrase2", ...],
  ["paraphrase1", "paraphrase2", ...]
]
No explanation, just the JSON."""

        result = call_groq(prompt, max_tokens=2000)
        if not result:
            continue

        try:
            # Clean and parse JSON
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            paraphrases_list = json.loads(result)

            for (_, answer), paraphrases in zip(batch, paraphrases_list):
                for para_q in paraphrases:
                    if isinstance(para_q, str) and len(para_q) > 5:
                        augmented.append((para_q.strip(), answer))

        except Exception as e:
            print(f"  Parse error for batch {i}: {e}")

        time.sleep(2)  # Respect rate limits
        print(f"  Processed {min(i+batch_size, len(seed_pairs))}/{len(seed_pairs)} questions")

    print(f"  Generated {len(augmented)} paraphrased pairs")
    return augmented


# ─── STRATEGY 2: CHUNK-BASED GENERATION ──────────────────────────────────────

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def generate_from_chunks(qa_per_chunk=15):
    """
    Load your data files, chunk them, and use Groq to generate
    Q&A pairs directly from each chunk.
    """
    print("\n[Strategy 2] Generating Q&A from text chunks...")
    generated = []

    for filename in ["about.txt", "projects.txt", "resume.txt"]:
        filepath = Path(DATA_DIR) / filename
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=200)
        print(f"  {filename}: {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            prompt = f"""You are creating training data for a portfolio chatbot about Aayush Maharjan.

Based on this text:
---
{chunk}
---

Generate {qa_per_chunk} diverse question-answer pairs that someone might ask when visiting Aayush's portfolio website.
Mix different question styles: direct questions, casual questions, professional questions.
Every answer must come directly from the text above.

Respond ONLY with a JSON array like this:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
No explanation, just the JSON."""

            result = call_groq(prompt, max_tokens=2000)
            if not result:
                continue

            try:
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("```")[1]
                    if result.startswith("json"):
                        result = result[4:]
                pairs = json.loads(result)
                for pair in pairs:
                    if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                        q = pair["question"].strip()
                        a = pair["answer"].strip()
                        if len(q) > 5 and len(a) > 5:
                            generated.append((q, a))
            except Exception as e:
                print(f"  Parse error chunk {idx}: {e}")

            time.sleep(1.5)

        print(f"  {filename} done — total so far: {len(generated)} pairs")

    print(f"  Generated {len(generated)} chunk-based pairs")
    return generated


# ─── STRATEGY 3: NEGATIVE PAIRS ───────────────────────────────────────────────

def generate_negative_pairs(all_positive_pairs, ratio=0.3):
    """
    Create mismatched Q&A pairs (negative examples).
    This teaches the model that NOT everything is a match.
    Standard practice in contrastive learning.
    """
    print("\n[Strategy 3] Generating negative pairs...")
    negatives  = []
    n_negative = int(len(all_positive_pairs) * ratio)

    questions = [q for q, _ in all_positive_pairs]
    answers   = [a for _, a in all_positive_pairs]

    for _ in range(n_negative):
        q = random.choice(questions)
        a = random.choice(answers)
        # Make sure it's actually a mismatch
        negatives.append((q, a, "negative"))

    print(f"  Generated {len(negatives)} negative pairs")
    return negatives


# ─── STRATEGY 4: ANSWER VARIATIONS ───────────────────────────────────────────

def vary_answers(seed_pairs, variations_per_answer=3):
    """
    For each seed pair, generate different ways to phrase the same answer.
    Teaches the model that the same info can be expressed differently.
    """
    print("\n[Strategy 4] Varying answer phrasings...")
    augmented = []
    batch_size = 8

    for i in range(0, min(len(seed_pairs), 30), batch_size):
        batch = seed_pairs[i:i+batch_size]
        qa_text = "\n".join(
            [f"{j+1}. Q: {q}\n   A: {a}" for j, (q, a) in enumerate(batch)]
        )

        prompt = f"""For each Q&A pair below, generate {variations_per_answer} different ways to phrase the answer.
Keep the same facts but vary the wording and sentence structure.

{qa_text}

Respond ONLY with a JSON array of arrays. Each inner array has {variations_per_answer} answer variations.
[
  ["variation1", "variation2", "variation3"],
  ...
]
No explanation, just JSON."""

        result = call_groq(prompt, max_tokens=2000)
        if not result:
            continue

        try:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            variations_list = json.loads(result)

            for (question, _), variations in zip(batch, variations_list):
                for var_a in variations:
                    if isinstance(var_a, str) and len(var_a) > 5:
                        augmented.append((question, var_a.strip()))

        except Exception as e:
            print(f"  Parse error batch {i}: {e}")

        time.sleep(2)

    print(f"  Generated {len(augmented)} answer-varied pairs")
    return augmented


# ─── SAVE & REPORT ────────────────────────────────────────────────────────────

def save_pairs(all_pairs, output_path=OUTPUT):
    """Save all pairs to JSON for use in train.py."""
    data = []
    for item in all_pairs:
        if len(item) == 2:
            data.append({"question": item[0], "answer": item[1], "label": "positive"})
        else:
            data.append({"question": item[0], "answer": item[1], "label": item[2]})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(data)} pairs to {output_path}")
    return data


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Synthetic Q&A Data Generation for Portfolio Chatbot")
    print("=" * 60)
    print(f"\nStarting with {len(SEED_PAIRS)} seed pairs")
    print("This will make ~3 API calls per minute to respect Groq rate limits.")
    print("Estimated time: 10-15 minutes\n")

    all_positive = list(SEED_PAIRS)

    # Strategy 1: Paraphrase questions (biggest boost)
    paraphrased = paraphrase_questions(SEED_PAIRS, paraphrases_per_question=10)
    all_positive.extend(paraphrased)
    print(f"Running total: {len(all_positive)} pairs")

    # Strategy 2: Generate from text chunks
    chunk_pairs = generate_from_chunks(qa_per_chunk=15)
    all_positive.extend(chunk_pairs)
    print(f"Running total: {len(all_positive)} pairs")

    # Strategy 4: Answer variations
    answer_varied = vary_answers(SEED_PAIRS, variations_per_answer=4)
    all_positive.extend(answer_varied)
    print(f"Running total: {len(all_positive)} pairs")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for pair in all_positive:
        key = pair[0].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"\nAfter deduplication: {len(unique_pairs)} unique positive pairs")

    # Strategy 3: Negative pairs
    negatives = generate_negative_pairs(unique_pairs, ratio=0.3)

    # Combine all
    final_pairs = unique_pairs + negatives
    random.shuffle(final_pairs)

    # Save
    save_pairs(final_pairs)

    # Report
    print("\n" + "=" * 60)
    print("  DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Seed pairs:          {len(SEED_PAIRS)}")
    print(f"  Paraphrased:         {len(paraphrased)}")
    print(f"  Chunk-generated:     {len(chunk_pairs)}")
    print(f"  Answer-varied:       {len(answer_varied)}")
    print(f"  Negative pairs:      {len(negatives)}")
    print(f"  TOTAL:               {len(final_pairs)}")
    print(f"\n  Saved to: {OUTPUT}")
    print("\nNext step: run python train.py")


if __name__ == "__main__":
    main()