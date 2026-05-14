"""
NEUST Handbook Chatbot — 

"""

import re
import math
import unicodedata
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from pyngrok import ngrok

# CONFIG

PDF_PATH             = "handbook.pdf"
MODEL_PATH           = "./neust-qa-bert-model"
CHUNK_SIZE           = 450
CHUNK_OVERLAP        = 80
TOP_K_BM25           = 12
TOP_K_FINAL          = 6
CONFIDENCE_THRESHOLD = 0.15  
MIN_ANSWER_LEN       = 2
MAX_ANSWER_LEN       = 200

app   = Flask(__name__)
_cache = {}

def qhash(q): return hashlib.md5(q.lower().strip().encode()).hexdigest()


# LOAD MODEL

print("[Model] Loading fine-tuned QA model from:", MODEL_PATH)
try:
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    model        = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    model.eval()
    model_loaded = True
    print("[Model] Ready.")
except Exception as e:
    print(f"[Model] Not loaded: {e}")
    model_loaded = False
    tokenizer = model = None


# PDF → TEXT → CHUNKS

def load_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(pages):
    chunks = []
    for p in pages:
        text      = clean_text(p["text"])
        sentences = re.split(r"(?<=[.!?]) +", text)
        current   = ""
        for s in sentences:
            if len(current) + len(s) <= CHUNK_SIZE:
                current += " " + s
            else:
                if current.strip():
                    chunks.append({"text": current.strip(), "page": p["page"]})
                overlap = " ".join(current.split()[-22:]) if current.split() else ""
                current = overlap + " " + s
        if current.strip():
            chunks.append({"text": current.strip(), "page": p["page"]})
    return chunks


# BM25 RETRIEVAL

STOPWORDS = set("a an the and or but is are was were to of in for on at by with from about as into through during before after above below be its that this which".split())

def tokenize(text):
    return [w.lower() for w in re.findall(r"\b\w+\b", text)
            if w.lower() not in STOPWORDS and len(w) > 1]

def build_idf(chunks):
    N, df = len(chunks), {}
    for c in chunks:
        for w in set(tokenize(c["text"])):
            df[w] = df.get(w, 0) + 1
    return {w: math.log((N - f + 0.5) / (f + 0.5) + 1) for w, f in df.items()}

def bm25_score(query, text, idf, avg_len, k1=1.5, b=0.75):
    q_toks = tokenize(query)
    d_toks = tokenize(text)
    if not d_toks: return 0.0
    tf = {}
    for t in d_toks: tf[t] = tf.get(t, 0) + 1
    doc_len = len(d_toks)
    score   = 0.0
    for q in q_toks:
        if q in tf:
            freq   = tf[q]
            score += idf.get(q, 0.5) * ((freq * (k1 + 1)) /
                     (freq + k1 * (1 - b + b * (doc_len / avg_len))))
    return score

def retrieve(question):
    if not KB: return []
    avg_len = sum(len(tokenize(c["text"])) for c in KB) / len(KB)
    bm25_scored = [(bm25_score(question, c["text"], IDF, avg_len), c) for c in KB]
    bm25_scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [c for _, c in bm25_scored[:TOP_K_BM25]]
    try:
        cand_texts = [c["text"] for c in candidates]
        cv   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat  = cv.fit_transform(cand_texts)
        qvec = cv.transform([question])
        cos  = cosine_similarity(qvec, mat).flatten()
        reranked = sorted(zip(cos, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in reranked[:TOP_K_FINAL]]
    except Exception:
        return candidates[:TOP_K_FINAL]


# MODEL INFERENCE —

def clean_answer(answer, question):
    """Post-process answer to extract clean, accurate text."""
    if not answer:
        return ""

    # Remove leading/trailing punctuation and spaces
    answer = answer.strip().strip('.,;:!?()[]{}"\'-')

    # If answer is too long, try to find a more concise match
    if len(answer) > 100:
        # Try to find the most relevant part that directly answers the question
        # Look for key terms in the question and find matching context
        q_keywords = set(w.lower() for w in question.split() if len(w) > 3)
        words = answer.split()
        best_part = answer
        best_score = 0

        # Find segment with most question keywords
        for i in range(len(words)):
            for j in range(i+3, min(i+20, len(words)+1)):
                part = ' '.join(words[i:j])
                part_words = set(w.lower() for w in part.split())
                score = len(q_keywords & part_words)
                if score > best_score and len(part) > 10:
                    best_score = score
                    best_part = part

        if best_score > 0:
            answer = best_part

    return answer.strip()

def answer_with_model(question, chunk):
    if not model_loaded: return "", 0.0
    try:
        max_ctx = 340
        ctx_toks = tokenizer.tokenize(chunk["text"])
        stride, windows = 120, []
        start = 0
        while start < len(ctx_toks):
            end = min(start + max_ctx, len(ctx_toks))
            windows.append(tokenizer.convert_tokens_to_string(ctx_toks[start:end]))
            if end == len(ctx_toks): break
            start += stride

        best_ans, best_score = "", 0.0
        all_answers = []  # Collect all candidate answers

        for win in windows:
            inputs = tokenizer(question, win, add_special_tokens=True,
                               return_tensors="pt", truncation="only_second",
                               max_length=384, padding="max_length")
            with torch.no_grad():
                out   = model(**inputs)
                s_log = out.start_logits[0]
                e_log = out.end_logits[0]
                s_prb = torch.softmax(s_log, dim=-1)
                e_prb = torch.softmax(e_log, dim=-1)
                ids   = inputs["input_ids"][0]

                # Use top-k more intelligently
                top_starts = s_prb.topk(15).indices
                top_ends = e_prb.topk(15).indices

                for si in top_starts:
                    for ei in top_ends:
                        if ei < si or ei - si > 40: continue  # Reduced max span length

                        sp = (s_prb[si] * e_prb[ei]).item()
                        if sp < 0.001: continue  # Skip very low probability spans

                        answer = tokenizer.decode(ids[si:ei+1],
                                                  skip_special_tokens=True).strip()

                        # Filter out invalid answers
                        if not answer or len(answer) < 2: continue
                        if any(t in answer for t in ["[CLS]","[SEP]","<s>","</s>"]): continue
                        if answer.lower() in question.lower(): continue

                        # Clean up the answer
                        answer = clean_answer(answer, question)

                        if len(answer) < MIN_ANSWER_LEN: continue
                        if len(answer) > MAX_ANSWER_LEN: continue

                        all_answers.append((answer, sp))

                        if sp > best_score:
                            best_score = sp
                            best_ans   = answer

        # Post-process: if best answer is too similar to question, try alternatives
        if best_ans:
            q_words = set(w.lower() for w in question.split())
            a_words = set(w.lower() for w in best_ans.split())

            # If answer is mostly question words, find better candidate
            if len(a_words) > 0 and len(q_words & a_words) / len(a_words) > 0.7:
                # Look for a less question-heavy answer
                for ans, sc in all_answers:
                    ans_words = set(w.lower() for w in ans.split())
                    overlap = len(q_words & ans_words) / max(len(ans_words), 1)
                    if overlap < 0.5 and sc > best_score * 0.5:
                        best_ans = ans
                        best_score = sc
                        break

        return best_ans, best_score
    except Exception as e:
        print(f"[Model] Inference error: {e}")
        return "", 0.0


# SENTENCE FALLBACK - improved

def find_best_sentence(question, chunks):
    sentences = []
    for chunk in chunks:
        for sent in re.split(r"(?<=[.!?])\s+", chunk["text"]):
            sent = sent.strip()
            # Include shorter sentences for Q&A (more likely to contain direct answers)
            if len(sent) > 20:
                sentences.append({"text": sent, "page": chunk["page"]})
    if not sentences: return "", None

    # Try TF-IDF matching first
    try:
        texts  = [s["text"] for s in sentences]
        vec    = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        mat    = vec.fit_transform(texts)
        q_vec  = vec.transform([question])
        scores = cosine_similarity(q_vec, mat).flatten()

        # Get top candidates
        top_indices = np.argsort(scores)[::-1][:5]
        for idx in top_indices:
            if scores[idx] > 0.08:  # Lower threshold for better recall
                return sentences[idx]["text"], sentences[idx]["page"]
    except Exception:
        pass

    # Fallback to keyword matching
    q_words = set(tokenize(question))
    best_s, best_p, best_sc = "", None, 0
    for s in sentences:
        s_words = set(tokenize(s["text"]))
        sc = len(q_words & s_words)
        # Prefer sentences with higher keyword density
        density = sc / max(len(s_words), 1)
        if sc > best_sc or (sc == best_sc and density > best_sc / max(len(set(tokenize(best_s))), 1)):
            best_sc, best_s, best_p = sc, s["text"], s["page"]

    if best_s and best_sc > 0:
        return best_s, best_p

    return "", None


# PATTERN MATCHING WITH PAGE NUMBERS
# ALL answers verified directly from handbook.pdf (Revised 2017)
# Format: (pattern, answer, page_number) — page may be omitted

PATTERNS = [

    # ── GENERAL INFO ──────────────────────────────────────────────
    (r"\btagline\b",
     "The NEUST tagline is: \"Nourishing the mind, Nurturing the heart, Leading the future.\"",
     1),

    (r"\bcore values?\b",
     "NEUST's core values are Nationalism, Excellence, Unity, Spirituality, and Transparency — forming the acronym NEUST.",
     1),

    (r"\bvision\b",
     "NEUST Vision: A globally renowned university as a champion of sustainable societal development through ethical and empowered human resources.",
     1),

    (r"\bmission\b",
     "NEUST Mission: Advanced knowledge generation and innovation, produce globally outstanding graduates, and transform communities towards inclusive progress.",
     1),

    (r"founded|wright institute|history of neust|when.*established|when.*start",
     "NEUST started in June 1908 as the Wright Institute. It was renamed Nueva Ecija Trade School (NETS) in 1929, then became NEUST later. Dr. Rhodora R. Jugo is the current University President.",
     1),

    (r"\bpresident\b|\bwho.*leads? neust\b|\bwho.*head",
     "The current University President of NEUST is Dr. Rhodora R. Jugo.",
     1),

    (r"how many campus|number of campus|list.*campus|campus.*location",
     "NEUST has six main campuses: General Tinio (Main Campus), Sumacab, San Isidro, Fort Magsaysay, Atate, and Gabaldon, plus off-campus extensions.",
     2),

    (r"philosophical statement|declaration of principles|students.?rights",
     "The NEUST Student Handbook outlines Students' Rights and Obligations, the Declaration of Principles, and the Philosophical Statement on Students under Part I — General Matters.",
     1),

    # ── ADMISSIONS ────────────────────────────────────────────────
    (r"admission requirement|requirements? for (first.?year|freshmen|new student|enrollment)|what.*need.*enroll",
     "First-year admission requirements: (1) GWA of at least 85% for board programs; (2) Pass the NEUST College Admission Test (NEUST CAT); (3) Form 138 (Report Card); (4) Certificate of Good Moral Character. "
     "Transferees need: GWA of 2.0, at least 33 units completed, and an Honorable Dismissal certificate.",
     10),

    (r"neust cat|college admission test|entrance exam",
     "The NEUST College Admission Test (NEUST CAT) is required for all first-year applicants seeking admission to NEUST.",
     10),

    (r"(minimum |required )?gwa.*(board|board program)|board.*gwa",
     "A minimum GWA of 85% (equivalent to a grade of 2.0) is required for admission to board programs at NEUST.",
     10),

    (r"foreign student|international student",
     "Foreign students must submit: a Student Visa, Alien Certificate of Registration (ACR), PHS, authenticated academic records, affidavit of support, and an English proficiency certificate.",
     10),

    (r"shift(ing)?.*program|shift(ing)?.*course|change.*course|transfer.*course",
     "A student may shift from a board to a non-board program upon the Dean's evaluation of his/her academic credentials. Shifting from a non-board to a board program requires a GWA of 1.75 and a Dean's recommendation.",
     11),

    (r"returnee|returning student|readmission",
     "Returning students (returnees) must have filed a Leave of Absence (LOA) and must attend the Freshmen Orientation upon return.",
     13),

    # ── GRADING SYSTEM ────────────────────────────────────────────
    (r"grading system|grade scale|how.*grade.*comput|grading policy",
     "NEUST Grading System:\n"
     "• 1.0 = 97–100% (Excellent)\n• 1.25 = 94–96%\n• 1.50 = 91–93%\n• 1.75 = 88–90%\n"
     "• 2.0 = 85–87%\n• 2.25 = 82–84%\n• 2.50 = 79–81%\n• 2.75 = 76–78%\n"
     "• 3.0 = 75% (Minimum passing)\n• 5.0 = 74% and below (Failed — requires re-enrollment)\n"
     "• INC = Incomplete (requirements not accomplished; must be removed within 1 year or becomes 5.0)\n"
     "• UD = Unofficially Dropped",
     25),

    (r"\binc\b.*grade|grade.*\binc\b|incomplete.*grade|what.*inc.*mean",
     "INC (Incomplete) is given to students with passing grades who failed to submit requirements. "
     "The deficiency must be removed within one (1) academic year; otherwise, it automatically becomes a grade of 5.0 (Failed). "
     "If the absence of examination is justifiable, the Dean may allow the student to take a make-up exam.",
     26),

    (r"passing grade|minimum passing|lowest passing grade",
     "The minimum passing grade at NEUST is 3.0, which corresponds to 75%.",
     25),

    (r"grade.*5\.0|5\.0.*grade|failing grade|what.*fail",
     "A grade of 5.0 means Failed (74% and below). The subject must be re-enrolled and repeated.",
     25),

    (r"unofficially dropped|ud grade|what.*ud",
     "UD (Unofficially Dropped) means the student did not submit an accomplished dropping form to the Registrar's Office. The student is marked UD in the subject.",
     26),

    (r"academic (warning|standing)|academic probation|probation|disqualif|retention policy",
     "Academic Standing Policies:\n"
     "• Academic Warning: final grade below 3.0 in subjects totaling 5 units in a semester.\n"
     "• Academic Probation: below 3.0 in subjects totaling 7 units; student load is limited.\n"
     "• Permanent Disqualification: 9 or more units of failing grades (3 subjects) in a board program; student is advised to transfer to a non-board program.",
     27),

    # ── GRADUATION & HONORS ───────────────────────────────────────
    (r"graduation requirement|qualify for graduation|how.*graduate|finish.*course",
     "To qualify for graduation, a student must: (1) Complete all required academic and non-academic units; "
     "(2) Clear all deficiencies at least 5 weeks before the end of the semester; "
     "(3) Have at least one year of residency at NEUST; and (4) Attend the General Commencement Exercises.",
     10),

    (r"summa cum laude",
     "Summa Cum Laude: GPA of 1.00 to 1.20. Requires 76% of units completed at NEUST, no grade below 2.0, and a minimum load of 15 units per semester.",
     29),

    (r"magna cum laude",
     "Magna Cum Laude: GPA of 1.21 to 1.45. Requires 76% of units completed at NEUST, no grade below 2.0, and a minimum load of 15 units per semester.",
     29),

    (r"\bcum laude\b(?!.*magna)(?!.*summa)",
     "Cum Laude: GPA of 1.46 to 1.75. Requires 76% of units completed at NEUST, no grade below 2.0, and a minimum load of 15 units per semester.",
     29),

    (r"latin honors|graduation (with )?honors|honor graduate|academic honors",
     "NEUST Latin Honors:\n"
     "• Summa Cum Laude: GPA 1.00–1.20\n"
     "• Magna Cum Laude: GPA 1.21–1.45\n"
     "• Cum Laude: GPA 1.46–1.75\n"
     "All require: at least 76% of units earned at NEUST, no grade below 2.0, and a minimum load of 15 units per semester.",
     29),

    (r"honorable dismissal",
     "Honorable Dismissal is issued for students voluntarily withdrawing from the University. "
     "All financial obligations must be cleared. If a student was dropped due to poor scholarship, a statement to that effect is added to the Honorable Dismissal.",
     10),

    # ── ATTENDANCE ────────────────────────────────────────────────
    (r"attendance policy|absence (policy|rule|limit)|how many absence|absent.*how many|maximum.*absent",
     "NEUST Attendance Policy: A student is ineligible for a passing grade if absent for the equivalent of 2.5 weeks:\n"
     "• 5 meetings/week: more than 12.5 absences\n"
     "• 4 meetings/week: more than 10 absences\n"
     "• 3 meetings/week: more than 7.5 absences\n"
     "• 2 meetings/week: more than 5 absences\n"
     "• 1 meeting/week: more than 2 absences\n"
     "Approved absences include: representing the University, sickness with physician's certificate, or parent/guardian excuse letter attested by the Dean.",
     13),

    (r"late|tardy|how.*late.*class|tardy.*policy",
     "A student is considered late or tardy if he/she arrives during the first one-third of the scheduled class time. "
     "For 60-minute classes, students must wait 20 minutes for a late professor before leaving; 30 minutes for 90-minute classes; 33 minutes for 100-minute classes; 40 minutes for 120-minute classes.",
     13),

    (r"leave of absence|loa\b",
     "A Leave of Absence (LOA) must be filed in writing, stating the reason and specifying the period, which shall not exceed one (1) academic year. "
     "The University will notify the Registrar and parent/guardian. Withdrawal without a formal LOA may result in cancellation of registration.",
     13),

    (r"dropping.*subject|drop.*subject|formally drop|ineligible.*passing",
     "Once a student exceeds the allowable number of absences, he/she becomes ineligible for a passing grade. "
     "The student must formally drop the subject by filing a Dropping Form with the Registrar's Office. Failure to do so will result in a failing grade.",
     13),

    # ── EXAMINATIONS ──────────────────────────────────────────────
    (r"exam.*schedule|midterm.*schedule|final.*exam.*schedule|when.*exam",
     "Midterm and final examination schedules are posted two (2) weeks before the examination period by the Office of the Registrar.",
     14),

    (r"exam.*exemption|exempt.*final|final exam.*exemp",
     "Faculty members may exempt students from final examinations, provided exemptions are not contrary to departmental policies. "
     "The faculty is under no obligation to grant exemptions. At the start of each semester, teachers inform students of their personal exemption policy.",
     14),

    (r"cheating.*penalty|penalty.*cheating|first.*cheat|cheat.*first|cheat.*offense|caught.*cheating",
     "Students caught cheating in any examination, test, or quiz shall be punished per Part V of the Handbook. "
     "Consequences include academic penalties and disciplinary action as determined by the Student Discipline Board.",
     90),

    # ── SCHOLARSHIPS (Fully verified from handbook pp. 61-70) ─────
    (r"what.*scholarship|scholarship.*available|list.*scholarship|kinds? of scholarship|types? of scholarship|all.*scholarship",
     "NEUST offers the following scholarship and financial assistance programs (per the Office of Scholarship and Student Financial Assistance):\n\n"
     "1. **Entrance Scholarship** — One semester only: Valedictorians receive full tuition; Salutatorians receive ½ tuition. "
     "Both must be from public/government-recognized Senior High Schools with a graduating class of at least 30 students.\n\n"
     "2. **Academic Incentive Scholarship** — Full: ₱4,000/semester for GPA of 1.5 or better (min. 15 units, no 5.0). "
     "Partial: ₱2,000/semester for GPA of 1.51–1.75.\n\n"
     "3. **Scholarship for Children/Legal Dependents of NEUST Personnel** — 100% free tuition and 50% miscellaneous fees "
     "(max. 3 children or spouse). Requires permanent appointment of the personnel.\n\n"
     "4. **Scholarship for NEUST Employees** — Permanent employees may apply for master's or doctoral scholarship "
     "(thesis/dissertation writing only, for 2 semesters).\n\n"
     "5. **Board of Regents Scholarship** — Each Regent sponsors one scholar at a time until graduation. "
     "Scholar pays ₱500 miscellaneous fee.\n\n"
     "6. **50% Tuition Discount for Cooperating Teachers** — For teachers handling NEUST practice teachers (1:2 ratio, 2 semesters).\n\n"
     "7. **USG Financial Assistance** — For lower-income students with good academic standing and no failing grades.\n\n"
     "8. **Cultural Groups Monetary Allowance** — Senior members: ₱5,000/semester; New members: ₱2,500/semester. "
     "Groups include Brass Band, Rondalla, Chorale, Danza, Folkloric, Theater, and more.\n\n"
     "9. **Sports (NEUST Phoenix) Monetary Allowance** — ₱1,000/month + ₱2,000 game allowance + ₱2,000 training allowance. "
     "Cash incentives: Gold ₱2,500, Silver ₱1,000, Bronze ₱800 (Regional/National competitions).\n\n"
     "10. **Student Leader Incentives** — USG President: ₱4,000/sem; VP: ₱3,200/sem; LSC Chairman: ₱2,800/sem; and others.\n\n"
     "11. **School Publication Staff** — Editor-in-Chief: ₱800–₱1,000/month; Associate Editor: ₱600–₱700/month.\n\n"
     "12. **Government Grants-in-Aid** — From CHED, DOST-SEI, and other government agencies.\n\n"
     "13. **Private Scholarships/Grants-in-Aid** — From private individuals, establishments, NGOs.",
     61),

    (r"entrance scholarship|valedictorian.*scholar|salutatorian.*scholar|scholar.*valedictorian",
     "Entrance Scholarship is granted for one (1) semester only:\n"
     "• Valedictorians: full cost of tuition fee\n"
     "• Salutatorians: ½ cost of tuition fee\n"
     "Requirements: graduate of a public/government-recognized Senior High School (Grade 12); "
     "graduating class of at least 30 students (principal's certification required); must satisfy college entrance requirements.",
     61),

    (r"academic incentive|incentive scholarship|academic scholarship",
     "Academic Incentive Scholarship:\n"
     "• Full: ₱4,000/semester — GPA of 1.5 or better in the preceding semester, with an academic load of at least 15 units and no grade of 5.0 in any subject.\n"
     "• Partial: ₱2,000/semester — GPA of 1.51 to 1.75, with at least 15 units and no grade of 5.0.\n"
     "(Source: Board Resolution No. 61, series 2018)",
     62),

    (r"scholarship.*children|scholarship.*dependent|scholarship.*personnel|children.*neust.*personnel|dependent.*scholarship",
     "Scholarship for Children/Legal Dependents of NEUST Personnel:\n"
     "• 100% free tuition fee and 50% miscellaneous (Trust Fund) and development fees (excluding student government and ID fees).\n"
     "• Granted to spouse and up to three (3) children until course completion.\n"
     "• Beneficiary must be enrolled at NEUST; not transferable.\n"
     "• Terminated upon separation, resignation, or retirement of the personnel.\n"
     "• Upon personnel's death while in service, scholarship continues until graduation of dependents.\n"
     "• Failing two minor subjects or one major subject allows one re-enrollment; failure again means forfeiture.\n"
     "(Source: Board Resolution No. 61, series 2007 and No. 100, s. 2002)",
     64),

    (r"scholarship.*employee|employee.*scholarship|faculty.*scholarship|staff.*scholarship",
     "Scholarship for NEUST Employees:\n"
     "• Teaching/non-teaching staff with temporary appointment enrolled in Master's or Doctoral programs are NOT eligible for scholarship but may receive financial assistance during thesis/dissertation writing.\n"
     "• Permanent employees may automatically apply for master's or doctoral scholarship.\n"
     "• Scholarship is awarded for thesis/dissertation writing for two (2) semesters only.\n"
     "• A certification from the Graduate School Dean is required.",
     65),

    (r"board of regents.*scholar|regent.*scholar|scholar.*regent",
     "Board of Regents Scholarship:\n"
     "• Each Regent is entitled to sponsor one (1) scholar at a time until graduation.\n"
     "• The Regent recommends the scholar in writing to the University President.\n"
     "• The scholar pays ₱500 for miscellaneous fees (plus insurance, ID, and SSG fees).\n"
     "• If the scholar stops, the Regent may recommend another the following semester.\n"
     "(Source: Board Resolution No. 53, series 2003)"),

    (r"usg.*financial|financial.*usg|lower.?income.*student|indigent.*student",
     "USG Financial Assistance is for lower-income students who:\n"
     "• Are bona fide students of NEUST;\n"
     "• Have good academic standing with no failing grades;\n"
     "• Are active participants in school activities or affiliated with a school organization; and\n"
     "• Belong to a lower-income family.\n"
     "(Source: Board Resolution No. 40, s. 2015)"),

    (r"cultural.*allowance|cultural.*incentive|cultural.*group.*scholar|brass band|rondalla|chorale|danza|theater group|folkloric|ethnic dance",
     "Cultural Groups Monetary Allowance (per semester):\n"
     "• Senior members (1 semester and above): ₱5,000/semester (₱1,000/month) or ₱10,000/school year\n"
     "• New members: ₱2,500/semester (₱500/month) or ₱7,500/school year\n"
     "Groups include: Brass Band (60 slots), Rondalla (15), Combo (15), Folkloric (25), Chorale (25), Danza (20), Theater Group (20), Ethnic Dance (20).\n"
     "Requirements: at least 15 enrolled units; no incomplete grade; no more than two grades of 5.0.\n"
     "(Source: Board Resolution No. 61, series 2018)"),

    (r"sports.*allowance|sports.*incentive|athlete.*allowance|phoenix.*allowance|athlete.*scholar",
     "NEUST Phoenix (Sports) Monetary Incentives:\n"
     "• Monthly allowance: ₱1,000/player\n"
     "• Game allowance: ₱2,000/player\n"
     "• Rigid training allowance: ₱2,000/player\n"
     "• Cash incentives for Regional/National competitions: Gold ₱2,500 | Silver ₱1,000 | Bronze ₱800\n"
     "(Source: Board Resolution No. 61, series 2018)"),

    (r"student leader.*incentive|incentive.*student leader|usg.*incentive|usg.*president.*incentive",
     "Student Leader Incentives (per semester):\n"
     "• USG President: ₱4,000\n"
     "• USG Vice President: ₱3,200\n"
     "• USG Executive Officers / Legislative & Judicial Chairman: ₱2,400\n"
     "• USG Officers / LSC Vice Chairman: ₱2,000\n"
     "• LSC Chairman: ₱2,800\n"
     "• LSC Secretary, Treasurer, Auditor: ₱1,600\n"
     "• LSC Business Manager, PIO, Representative: ₱1,200\n"
     "(Source: Board Resolution No. 61, series 2018)"),

    (r"school publication.*incentive|editor.*incentive|publication staff|editor.?in.?chief",
     "School Publication Staff Monthly Incentives:\n"
     "• Editor-in-Chief: ₱800–₱1,000/month\n"
     "• Associate Editor: ₱600–₱700/month\n"
     "• Managing Editor: ₱300–₱600/month\n"
     "Subject to efficient performance and availability of funds."),

    (r"cooperating teacher.*discount|50%.*tuition.*cooperating|cooperating.*50",
     "50% Tuition Fee Discount for Cooperating Teachers:\n"
     "• Teachers from cooperating schools who handle NEUST practice teachers receive 50% tuition discount.\n"
     "• Ratio is 1:2 — for every semester of handling practice teachers, the Cooperating Teacher enjoys the discount for two (2) semesters.\n"
     "(Source: Board Resolution No. 81, s. 2008)"),

    (r"government.*scholarship|ched.*scholarship|dost.*scholarship|government.*grant",
     "Government Scholarship/Grant-in-Aid Programs at NEUST include those administered by:\n"
     "• CHED (Commission on Higher Education) — various programs\n"
     "• DOST-SEI (Department of Science and Technology – Science Education Institute)\n"
     "• Local and national government officials for their constituents.\n"
     "Inquire at the Office of Scholarship and Student Financial Assistance Program for current availability."),

    (r"private scholarship|ngo.*scholarship|private.*grant",
     "Private Scholarship/Financial Grants-in-Aid Programs are given by private individuals, establishments, institutions, and non-governmental organizations (NGOs). "
     "The Office of Scholarship may assist in identifying potential recipients for donations, employment, and scholarship opportunities from private sources."),

    # ── STUDENT LOANS ─────────────────────────────────────────────
    (r"student loan|registration loan|emergency loan|loan.*program|types? of loan",
     "NEUST Student Loan Assistance Program offers the following types of loans:\n"
     "• **Registration Loan** — Maximum: amount of assessed fees for enrollment. Interest: 4% per semester; surcharge: 1%/month if unpaid.\n"
     "• **Emergency Loan** — Max ₱200/semester (or ₱50/month). Interest: 1%/month; surcharge: 0.5%/month if delayed.\n"
     "• **Food Allowance Loan** — For students unable to meet food needs due to calamities.\n"
     "• **Dormitory/Lodging Loan** — For urgent lodging fee payment.\n"
     "• **Sickness Loan** — For students who fall ill and need immediate medication.\n"
     "• **Project Loan** — For expenses of required projects in enrolled subjects.\n"
     "• **Thesis Loan** — To fund thesis completion (including research papers in undergraduate level).\n"
     "• **Field/Educational Trip Loan** — For approved field trips listed in the subject syllabus.\n"
     "Failure to settle any loan by the due date results in no loan privileges the following semester. No grades, clearances, TOR, or diplomas will be released to students with unpaid loans."),

    (r"registration loan",
     "Registration Loan: Extended to needy but deserving students. Maximum amount is equivalent to the assessed enrollment fee. "
     "Charged a simple interest of 4% per semester and a surcharge of 1% per month or fraction thereof if unpaid on the due date."),

    (r"emergency loan",
     "Student Emergency Loan: Extended to all registered students for emergency needs (transportation, contributions, urgent expenses). "
     "Maximum: ₱200/semester or ₱50/month. Charged 1% interest per month and 0.5% surcharge per month if payment is delayed."),

    # ── LIBRARY ───────────────────────────────────────────────────
    (r"library hours|library.*open|open.*library|what time.*library",
     "Library Hours:\n• Monday–Friday: 7:30 AM – 6:30 PM\n• Saturday: 8:00 AM – 12:00 PM and 1:00 PM – 5:00 PM"),

    (r"borrow.*book|book.*borrow|library.*limit|how many.*book|books.*library",
     "Library Borrowing Rules:\n"
     "• Circulation Section: up to 3 books for 1 week\n"
     "• Filipiniana Section: 1 book overnight\n"
     "A Borrower's Card is required. Present your Student ID and Certificate of Registration to get one."),

    (r"library card|borrower.?s card",
     "A Borrower's Card is required to borrow books from the library. Present your Student ID and Certificate of Registration to the library."),

    (r"lost.*book|damaged.*book|library.*fine",
     "A lost or damaged library book must be paid for or replaced with the latest edition of the same title."),

    # ── STUDENT SERVICES ──────────────────────────────────────────
    (r"medical exam|health exam|physical exam|health requirement",
     "All enrolling students undergo: physical examination, chest X-ray, CBC (Complete Blood Count), urinalysis, and blood typing. "
     "Limited dental services (extraction, restoration, examination) are also available."),

    (r"guidance (counseling|office)|counseling.*office|guidance.*service",
     "The NEUST Guidance Office offers: personal counseling, career guidance, mental health programs, peer facilitation, personality assessment, and aptitude testing. "
     "The Office of Testing also administers the NEUST College Admission Test."),

    (r"nstp|rotc|national service training",
     "NSTP (National Service Training Program) is required for all baccalaureate degree students. It covers civic welfare and related programs."),

    (r"student id|id card|lost.*id",
     "Student IDs are issued by the Registrar's Office and must be validated each semester. A lost ID must be reported immediately to the Registrar's Office."),

    (r"transcript of records|\btor\b",
     "The Official Transcript of Records (TOR) is issued for transfer purposes. It requires clearance and payment of the applicable fee. "
     "No TOR will be released to students with unpaid loans."),

    # ── STUDENT ORGANIZATIONS ────────────────────────────────────
    (r"student organization.*recogni|recognize.*organization|apply.*organization|form.*organization",
     "To apply for recognition, a group of at least 15 students must submit: a Constitution and By-Laws, a ₱300 application fee, and a designated faculty adviser to the Office of Student Organizations, Activities and Development (OSOAD)."),

    (r"accredited.*org|accreditation.*org|accredited.*student",
     "Organizations may apply for accreditation after 5 consecutive years of recognition. Accredited organizations are exempt from yearly renewal for 3 to 5 years."),

    (r"fund.?rais",
     "Recognized organizations may conduct fundraising activities with a permit from the appropriate office. The maximum net target is ₱10,000."),

    # ── DISCIPLINE ────────────────────────────────────────────────
    (r"grounds.*discipline|disciplinary action|offenses|misconduct|student.*violat",
     "Grounds for discipline include: dishonesty, cheating, theft, oppression, misconduct, drug/alcohol violation within the campus, vandalism of school property, and insubordination."),

    (r"suspension.*penalty|expulsion|penalty.*suspension|disciplinary.*penalty",
     "Disciplinary penalties range from suspension (1 week to 2 semesters) to permanent expulsion for severe offenses such as threats to life, violence, or criminal acts."),

    (r"student judicial|sjc\b",
     "The Student Judicial Council (SJC) handles complaints among students. It is composed of 5 students appointed by the University President."),

    (r"student discipline board|discipline board",
     "The Student Discipline Board handles complaints filed against students. It includes an arbiter (lawyer), faculty representative, administration representative, and student representatives."),

    # ── GAD & SPECIAL OFFICES ─────────────────────────────────────
    (r"gad office|gender and development|magna carta.*women",
     "The GAD (Gender and Development) Office is located at the General Tinio Campus. It implements the Magna Carta for Women and is directed by CHED."),

    (r"ojt|on.the.job training|practicum",
     "OJT (On-the-Job Training) is required for graduating students. It is supervised by the Host Training Establishment (HTE) and a NEUST University coordinator. Applicable fees apply."),

    (r"off.?campus.*activit|field trip|educational tour|educational trip",
     "Off-campus activities require: parental consent, medical clearance, insurance, and CHED compliance. "
     "A Certificate of Compliance must be submitted to the CHEDRO at least 15 days before the activity."),

    # ── GRADUATE SCHOOL ───────────────────────────────────────────
    (r"graduate school admission|master.?s.*admission|master.?s.*program.*enter",
     "Master's programs at NEUST require: a baccalaureate degree with an average of 85%, and passing the English proficiency test."),

    (r"doctorate|phd.*admission|doctoral.*admission",
     "Doctorate programs require: a master's degree with a weighted average of at least 1.75."),

    (r"thesis|dissertation",
     "A thesis is required for master's degrees; a dissertation is required for doctorate programs. Both require a defense panel and applicable fees."),
]


# PATTERN MATCHING FUNCTION (handles both 2‑ and 3‑element tuples)

def pattern_match(question):
    """Returns (answer, page_number) tuple or None. Handles both (pattern, answer) and (pattern, answer, page)."""
    q = question.lower().strip()
    for pattern_data in PATTERNS:
        if len(pattern_data) == 3:
            pattern, answer, page = pattern_data
        else:
            pattern, answer = pattern_data
            page = None
        if re.search(pattern, q, re.IGNORECASE):
            return answer, page
    return None


# MAIN PIPELINE

def get_answer(question):
    key = qhash(question)
    if key in _cache:
        return {**_cache[key], "from_cache": True}

    # 1 — Pattern matching (handbook-verified answers)
    pat = pattern_match(question)
    if pat:
        answer, page = pat
        result = {
            "reply": answer,
            "pages": [page] if page else [],
            "score": 0.95,
            "confidence": "high",
            "has_answer": True,
            "from_cache": False
        }
        _cache[key] = result
        return result

    # 2 — Hybrid retrieval
    chunks = retrieve(question)
    if not chunks:
        return _no_answer()

    # 3 — Fine-tuned model with improved inference
    if model_loaded:
        best_ans, best_score, best_pages = "", 0.0, []
        for chunk in chunks:
            ans, score = answer_with_model(question, chunk)
            if ans and score > best_score:
                best_score = score
                best_ans   = ans
                best_pages = [chunk["page"]]

        # Accept answer if it meets threshold OR has reasonable confidence
        if best_ans:
            # Additional validation: check answer is relevant to question
            q_words = set(w.lower() for w in question.split() if len(w) > 2)
            a_words = set(w.lower() for w in best_ans.split() if len(w) > 2)

            # Don't reject if answer has enough content even with high overlap
            is_valid = True
            if best_score < CONFIDENCE_THRESHOLD and len(a_words) > 0:
                # If score is low but answer exists, check quality
                # Accept if answer is substantial and not just question repetition
                if len(best_ans) < 10:  # Too short
                    is_valid = False

            if is_valid and (best_score >= CONFIDENCE_THRESHOLD or len(best_ans) > 15):
                conf = "high" if best_score > 0.50 else "medium"
                result = {
                    "reply": best_ans,
                    "pages": best_pages,
                    "score": round(best_score, 4),
                    "confidence": conf,
                    "has_answer": True,
                    "from_cache": False
                }
                _cache[key] = result
                return result

    # 4 — Sentence fallback with improved matching
    sent, page = find_best_sentence(question, chunks)
    if sent:
        # Validate the sentence is relevant
        q_words = set(tokenize(question))
        a_words = set(tokenize(sent))
        if len(q_words & a_words) > 0:
            result = {
                "reply": sent,
                "pages": [page] if page else [],
                "score": 0.5,
                "confidence": "medium",
                "has_answer": True,
                "from_cache": False
            }
            _cache[key] = result
            return result

    return _no_answer()

def _no_answer():
    return {
        "reply": "I'm sorry, I couldn't find a specific answer to that question in the NEUST Student Handbook. "
                 "Please try rephrasing your question, or contact the Registrar's Office or the relevant department for assistance.",
        "pages": [],
        "score": 0.0,
        "confidence": "low",
        "has_answer": False,
        "from_cache": False
    }


# INIT

print("[KB] Loading handbook PDF...")
pages          = load_pdf(PDF_PATH)
KB             = chunk_text(pages)
IDF            = build_idf(KB)
print(f"[KB] {len(KB)} chunks loaded.")


# FLASK ROUTES

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({
            "reply": "Please type a question.",
            "pages": [],
            "score": 0.0,
            "confidence": "low",
            "has_answer": False
        })
    return jsonify(get_answer(question))

@app.route("/status")
def status():
    return jsonify({
        "chunks": len(KB),
        "model_loaded": model_loaded,
        "patterns": len(PATTERNS),
        "cache_size": len(_cache)
    })

if __name__ == "__main__":
    # Try to start ngrok tunnel, but continue without it if it fails
    #public_url = None
    #try:
        #public_url = ngrok.connect(5000)
        #print(f"[Ngrok] Public URL: {public_url}")
    #except Exception as e:
        #print(f"[Ngrok] Skipped (not configured or failed: {e})")

    print("\n" + "="*50)
    print(" - NEUST HANDBOOK CHATBOT -")
    print("="*50)
    print(f"  Local: http://localhost:5000")
    #if public_url:
        #print(f"  Public URL: {public_url}")
    print(f"  Model: {'Loaded ✓' if model_loaded else 'Pattern+Fallback only'}")
    print(f"  Patterns: {len(PATTERNS)} verified from handbook.pdf")
    print(f"  Chunks: {len(KB)}")
    print("="*50 + "\n")

    # Try to open browser automatically
    try:
        import webbrowser
        webbrowser.open("http://localhost:5000")
    except:
        pass

    app.run(debug=False, host="0.0.0.0", port=5000)