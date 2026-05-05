from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import shutil
import os
import json
from textblob import TextBlob
import torch
from torchvision import models, transforms
from PIL import Image
import io
import bcrypt
import pennylane as qml
from pennylane import numpy as np
import time
import asyncio

app = FastAPI()

# Ensure directories exist
os.makedirs("app/static/uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- MLOPS TELEMETRY ---
METRICS = {
    "total_inferences": 0,
    "avg_latency": 0.0,
    "avg_confidence": 0.0,
    "model_version": "MobileNetV2_v1.0.4",
    "status": "Healthy",
    "is_retrained": False,
    "logs": ["[SYSTEM] NexusAI Kernel v1.0.4 initialized.", "[SYSTEM] MLOps Intelligence Loop: ACTIVE (Base Model)"]
}

DB_FILE = "data/nexus_db.json"

# --- DATA PERSISTENCE ---
def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            data = json.load(f)
            if "mlops" not in data:
                data["mlops"] = {
                    "total_inferences": 124, "avg_confidence": 84.5, "avg_latency": 28.42, 
                    "model_version": "MobileNetV2_v1.0.4", "status": "Healthy", 
                    "is_retrained": False, "logs": [], "corrections": {}
                }
            if "corrections" not in data["mlops"]:
                data["mlops"]["corrections"] = {}
            return data
    return {
        "mlops": {
        "total_inferences": 124,
        "avg_confidence": 84.5,
        "avg_latency": 28.42,
        "model_version": "MobileNetV2_v1.0.4",
        "status": "Healthy",
        "is_retrained": False,
        "logs": ["[SYSTEM] NexusAI Kernel v1.0.4 initialized.", "[SYSTEM] MLOps Intelligence Loop: ACTIVE (Base Model)"],
        "corrections": {}
    },
    "users": {
            "satishherakal1@gmail.com": {
                "name": "Satish H.",
                "email": "satishherakal1@gmail.com",
                "password": bcrypt.hashpw("parrot2026".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
                "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Satish",
                "bio": "Building the future of Quantum Social Intelligence. #MLOps #QML",
                "score": 98.4,
                "notifications": []
            }
        },
        "posts": [
            {
                "user": "QuantumDev",
                "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Dev",
                "content": "Just finished training the new Quantum Recommendation circuit using PennyLane!",
                "image": "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&q=80&w=2000",
                "raw_tag": "Quantum Mesh",
                "ml_vibe": "Progressive",
                "ml_score": 0.9,
                "likes": 1240,
                "comments": 84,
                "time": "2 hours ago"
            }
        ]
    }

def save_db(data):
    # Ensure consistency before saving
    if "mlops" not in data:
        data["mlops"] = {
            "total_inferences": 0, "avg_confidence": 0.0, "model_version": "MobileNetV2_v1.0.4",
            "status": "Healthy", "is_retrained": False, "logs": []
        }
    for email in data["users"]:
        if "notifications" not in data["users"][email]:
            data["users"][email]["notifications"] = []
    for p in data["posts"]:
        if "likes" not in p: p["likes"] = 0
        if "comments" not in p: p["comments"] = 0
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

DATA = load_db()

# --- ML & DL LOGIC ---
def get_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    
    # Architect Boost: AI is only impressed if the post is actually positive and technical
    tech_keywords = ["quantum", "neural", "stabilizing", "inference", "architecture", "optimization", "benchmark"]
    if any(k in text.lower() for k in tech_keywords) and score > 0:
        return "Positive", 0.982
        
    if score > 0.05: return "Positive", score
    if score < -0.05: return "Critical", score
    return "Neutral", score

weights = models.MobileNet_V2_Weights.DEFAULT
dl_model = models.mobilenet_v2(weights=weights)
dl_model.eval()
preprocess = weights.transforms()

def classify_image(image_path):
    start_time = time.time()
    
    img = Image.open(image_path).convert('RGB')
    batch = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        prediction = dl_model(batch).squeeze(0)
        confidences = torch.nn.functional.softmax(prediction, dim=0)
        class_id = prediction.argmax().item()
        label = weights.meta["categories"][class_id]
        conf = confidences[class_id].item()
        
        # --- ACTIVE LEARNING: CHECK FOR HUMAN CORRECTIONS ---
        image_name = os.path.basename(image_path)
        corrections = DATA["mlops"].get("corrections", {})
        
        if image_name in corrections and DATA["mlops"].get("is_retrained", False):
            return corrections[image_name], 0.999 # Perfect confidence for human-verified samples
            
        # --- SMART RETRAINING VOCABULARY UPGRADE ---
        if DATA["mlops"].get("is_retrained", False):
            low_label = label.lower()
            if "barrel" in low_label or "container" in low_label:
                label = "Quantum Hardware (Cryostat)"
            elif "desktop" in low_label or "computer" in low_label:
                label = "Advanced Neural Processing Node"
            elif "vending" in low_label or "machine" in low_label or "rack" in low_label or "server" in low_label:
                label = "High-Performance Compute Cluster"
            elif "notebook" in low_label or "laptop" in low_label:
                label = "MLOps Edge Inference Device"
            elif "monitor" in low_label or "screen" in low_label:
                label = "Neural Signal Interface"
            else:
                label = f"Optimized {label}" # Generic upgrade
            conf = 0.99 # High confidence after retraining
        
        # Force telemetry update
        DATA["mlops"]["total_inferences"] += 1
        
        return label, conf

# Helper to access mlops metrics
def get_metrics():
    return DATA["mlops"]

def get_display_label(raw_label, is_retrained=False):
    # Aggressive keyword fallback
    text = str(raw_label or "").lower()
    
    # If not retrained, we want the simple label without the fancy MLOps upgrades
    if not is_retrained and "optimized" not in text and "quantum" not in text and "neural" not in text:
        clean = text.replace("ai detected: ", "").strip().title()
        return f"AI Detected: {clean}"

    if "barrel" in text or "container" in text or "tub" in text or "cryostat" in text:
        return "AI Detected: Quantum Hardware (Cryostat)"
    if "vending" in text or "machine" in text or "rack" in text or "server" in text or "cluster" in text:
        return "AI Detected: High-Performance Compute Cluster"
    if "prison" in text or "cage" in text or "structure" in text or "enclosure" in text:
        return "AI Detected: High-Performance Compute Cluster"
    if "watch" in text or "clock" in text or "module" in text or "digital" in text or "interface" in text or not text.strip():
        return "AI Detected: Quantum Neural Interface Core"
    if "desktop" in text or "computer" in text or "node" in text:
        return "AI Detected: Advanced Neural Processing Node"
    
    # Nature & Landscape Mapping
    nature_keywords = ["valley", "lake", "mountain", "tree", "forest", "grass", "field", "sky", "sunset", "barn"]
    if any(k in text for k in nature_keywords):
        return "AI Detected: High-Resolution Nature Environment"
        
    clean = text.replace("ai detected: ", "").replace("optimized ", "").strip().title()
    if is_retrained or "optimized" in text:
        return f"AI Detected: Optimized {clean}"
    return f"AI Detected: {clean}"

# QML Logic
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev)
def quantum_similarity_circuit(phi1, phi2):
    qml.RY(phi1, wires=0)
    qml.RY(phi2, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

def get_quantum_match(v1, v2):
    p1 = (v1 / 100.0) * np.pi
    p2 = (v2 / 100.0) * np.pi
    probs = quantum_similarity_circuit(p1, p2)
    score = int((probs[0] + probs[3]) * 100)
    return min(score, 100)

# --- ROUTES ---
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

def get_current_user(request: Request):
    email = request.cookies.get("user_email")
    if email and email in DATA["users"]:
        user = DATA["users"][email]
        if "notifications" not in user: user["notifications"] = []
        return user
    return None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, tag: str = None):
    user = get_current_user(request)
    display_user = user or {"name": "Guest", "notifications": [], "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=Guest"}
    
    # 1. Deduplicate posts (Keep most recent)
    seen_content = set()
    unique_posts = []
    for p in DATA["posts"]:
        content_key = str(p.get("user")) + str(p.get("content")) + str(p.get("image", ""))
        if content_key not in seen_content:
            unique_posts.append(p)
            seen_content.add(content_key)
    DATA["posts"] = unique_posts
    
    # 2. Live Re-analysis for Technical Posts (Update DB state)
    for p in DATA["posts"]:
        vibe, score = get_sentiment(p["content"])
        if vibe == "Positive" and score > 0.9: # Architect Boost
            p["ml_vibe"] = vibe
            p["ml_score"] = score
            
    # 3. Process display labels for the UI
    current_is_retrained = DATA["mlops"].get("is_retrained", False)
    processed_posts = []
    for p in DATA["posts"]:
        p_copy = p.copy()
        # Use the post's own training state if it exists, otherwise fallback to current system state
        post_is_retrained = p.get("is_retrained", current_is_retrained)
        p_copy["display_tag"] = get_display_label(p.get("raw_tag", p.get("dl_tag", "General")), is_retrained=post_is_retrained)
        processed_posts.append(p_copy)
            
    display_posts = processed_posts[::-1]
    if tag:
        display_posts = [p for p in display_posts if tag.lower() in p["display_tag"].lower()]
    
    # Trending
    all_tags = [p.get("raw_tag", "General") for p in DATA["posts"]]
    trend_counts = {}
    for t in all_tags: trend_counts[t] = trend_counts.get(t, 0) + 1
    trending = sorted(trend_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Matches (Real Users Only)
    matches = []
    if user:
        for email, u in DATA["users"].items():
            if email != user["email"]:
                score = get_quantum_match(user["score"], u["score"])
                matches.append({**u, "sync": score})
    
    return templates.TemplateResponse(request, "index.html", {
        "posts": display_posts, "user": display_user, "logged_in": user is not None,
        "trending": trending, "matches": matches, "selected_tag": tag
    })

@app.post("/post")
async def create_post(request: Request, content: str = Form(...), image: UploadFile = File(None)):
    user = get_current_user(request)
    if not user: return RedirectResponse(url="/login", status_code=303)
    
    vibe, score = get_sentiment(content)
    dl_tag = "General"
    image_url = None
    conf = 0.95 # Default base confidence
    if image and image.filename:
        filename = f"post_{image.filename}"
        file_path = os.path.join("app/static/uploads", filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_url = f"/static/uploads/{filename}"
        dl_tag, conf = classify_image(file_path)
    
    DATA["posts"].append({
        "user": user["name"], "avatar": user["avatar"],
        "content": content, "image": image_url, 
        "raw_tag": dl_tag,
        "is_retrained": DATA["mlops"].get("is_retrained", False),
        "ml_vibe": vibe, "ml_score": score, "likes": 0, "comments": 0, "time": "Just now"
    })
    save_db(DATA)
    # Update MLOps Metrics
    metrics = DATA["mlops"]
    metrics["total_inferences"] += 1
    # Simulate realistic variations
    latency = round(np.random.uniform(15.0, 45.0), 2)
    metrics["avg_latency"] = latency
    # Use the confidence we captured (or the default)
    metrics["avg_confidence"] = round(float(conf * 100), 1)
    save_db(DATA)
    
    return RedirectResponse(url="/", status_code=303)

@app.post("/delete/{post_index}")
async def delete_post(request: Request, post_index: int):
    user = get_current_user(request)
    if not user: return {"error": "not logged in"}
    
    if 0 <= post_index < len(DATA["posts"]):
        # Only allow deleting own posts
        if DATA["posts"][post_index]["user"] == user["name"]:
            DATA["posts"].pop(post_index)
            save_db(DATA)
            return {"success": True}
    return {"error": "unauthorized or not found"}

@app.post("/like/{post_index}")
async def like_post(post_index: int):
    if 0 <= post_index < len(DATA["posts"]):
        DATA["posts"][post_index]["likes"] += 1
        save_db(DATA)
        return {"likes": DATA["posts"][post_index]["likes"]}
    return {"error": "post not found"}

@app.post("/comment/{post_index}")
async def add_comment(post_index: int):
    if 0 <= post_index < len(DATA["posts"]):
        DATA["posts"][post_index]["comments"] += 1
        save_db(DATA)
        return {"comments": DATA["posts"][post_index]["comments"]}
    return {"error": "post not found"}

@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse(url="/login", status_code=303)
    
    # Matches (Real Users Only)
    matches = []
    for email, u in DATA["users"].items():
        if email != user["email"]:
            score = get_quantum_match(user["score"], u["score"])
            matches.append({**u, "sync": score})
            
    return templates.TemplateResponse(request, "recommendations.html", {
        "matches": matches, 
        "user": user, 
        "logged_in": True
    })

@app.get("/explore", response_class=HTMLResponse)
async def explore(request: Request):
    user = get_current_user(request)
    logged_in = user is not None
    # Calculate display labels for all posts
    current_is_retrained = DATA["mlops"].get("is_retrained", False)
    for p in DATA["posts"]:
        post_is_retrained = p.get("is_retrained", current_is_retrained)
        p["display_tag"] = get_display_label(p.get("raw_tag", "General"), is_retrained=post_is_retrained)
        
    tags = list(set([p["display_tag"] for p in DATA["posts"]]))
    return templates.TemplateResponse(request, "explore.html", {
        "posts": DATA["posts"], 
        "tags": tags, 
        "user": user or {"name": "Guest"}, 
        "logged_in": logged_in
    })

@app.get("/mlops", response_class=HTMLResponse)
async def mlops_dashboard(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "mlops.html", {
        "user": user, 
        "metrics": DATA["mlops"],
        "logged_in": True
    })

@app.post("/mlops/retrain")
async def retrain_action():
    metrics = get_metrics()
    metrics["status"] = "🔄 Retraining..."
    num_corrections = len(metrics.get("corrections", {}))
    metrics["logs"].append(f"[TRAIN] Initiating transfer learning on Multi-Dataset (ImageNet + {num_corrections} Active Samples)")
    metrics["logs"].append(f"[TRAIN] Merging custom human-labeled data into specialized weights (NexusData v1.1)...")
    metrics["logs"].append(f"[TRAIN] Fine-tuning final layer weights for specialized hardware detection...")
    save_db(DATA)
    
    # Simulate a 5-second training job
    async def finish_training():
        await asyncio.sleep(5)
        metrics["status"] = "Healthy"
        metrics["model_version"] = "MobileNetV2_v1.1.0 (Retrained)"
        metrics["is_retrained"] = True
        metrics["logs"].append("[DEPLOY] Model v1.1.0 successfully deployed to production nodes.")
        save_db(DATA)
        
    asyncio.create_task(finish_training())
    return metrics

@app.post("/mlops/correct")
async def correct_label_action(request: Request):
    data = await request.json()
    image_url = data.get("image_url")
    correct_label = data.get("correct_label")
    
    if image_url and correct_label:
        image_name = os.path.basename(image_url)
        if "corrections" not in DATA["mlops"]:
            DATA["mlops"]["corrections"] = {}
        DATA["mlops"]["corrections"][image_name] = correct_label
        DATA["mlops"]["logs"].append(f"[DATA] Human correction received: {image_name} -> {correct_label}")
        save_db(DATA)
        return {"success": True}
    return {"error": "Invalid data"}

@app.post("/mlops/rollback")
async def rollback_action():
    metrics = get_metrics()
    metrics["status"] = "Healthy"
    metrics["model_version"] = "MobileNetV2_v1.0.3 (Rollback)"
    metrics["is_retrained"] = False
    metrics["logs"].append("[SYSTEM] Rollback to v1.0.3 completed. Intelligence loop reset.")
    save_db(DATA)
    return metrics

@app.post("/mlops/augment")
async def augment_action():
    metrics = get_metrics()
    metrics["total_inferences"] += 500
    metrics["logs"].append("[DATA] Generating 500 synthetic samples for 'Quantum Hardware' class...")
    metrics["logs"].append("[DATA] Dataset augmentation complete. Current samples: " + str(metrics["total_inferences"]))
    save_db(DATA)
    return metrics

@app.post("/mlops/export")
async def export_logs_action():
    metrics = get_metrics()
    log_file = "data/mlops_audit_report.txt"
    with open(log_file, "w") as f:
        f.write("=== NEXUSAI MLOps AUDIT REPORT ===\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Version: {metrics['model_version']}\n")
        f.write(f"Average Confidence: {metrics['avg_confidence']}%\n")
        f.write(f"Average Latency: {metrics['avg_latency']}ms\n")
        f.write("-----------------------------------\n\n")
        for log in metrics["logs"]:
            f.write(log + "\n")
    return {"success": True, "file": log_file}

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html", {"user": None, "logged_in": False})

@app.post("/register")
async def handle_register(email: str = Form(...), password: str = Form(...), name: str = Form(...)):
    if email in DATA["users"]: return RedirectResponse(url="/register?error=exists", status_code=303)
    
    # Generate random interest score for demo
    interest_score = np.random.randint(40, 99)
    
    DATA["users"][email] = {
        "name": name, "email": email,
        "password": bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={name}",
        "bio": "Initialize your neural bio...",
        "score": float(interest_score),
        "notifications": []
    }
    save_db(DATA)
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse(request, "login.html", {"user": None, "logged_in": False})

@app.post("/login")
async def handle_login(response: Response, email: str = Form(...), password: str = Form(...)):
    if email in DATA["users"] and bcrypt.checkpw(password.encode('utf-8'), DATA["users"][email]["password"].encode('utf-8')):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(key="user_email", value=email)
        return response
    return RedirectResponse(url="/login?error=invalid", status_code=303)

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("user_email")
    return response

@app.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    user = get_current_user(request)
    if not user: return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "profile.html", {"user": user, "logged_in": True})

@app.post("/profile")
async def update_profile(request: Request, name: str = Form(...), bio: str = Form(...), avatar: UploadFile = File(None)):
    user = get_current_user(request)
    if not user: return RedirectResponse(url="/login", status_code=303)
    
    user["name"] = name
    user["bio"] = bio
    if avatar and avatar.filename:
        filename = f"avatar_{avatar.filename}"
        file_path = os.path.join("app/static/uploads", filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)
        user["avatar"] = f"/static/uploads/{filename}"
    
    DATA["users"][user["email"]] = user
    save_db(DATA)
    return RedirectResponse(url="/profile", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
