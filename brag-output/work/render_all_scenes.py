"""
NEXUS-OPS PS-Q1 Full Video Frame Renderer
Generates 600 frames (20 seconds @ 30fps) at 1920x1080 resolution.
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
FPS = 30
TOTAL_FRAMES = 600

# Color Palette
BG_COLOR = (7, 10, 20)           # Deep Quantum Navy
NAVY_CARD = (14, 21, 38)
CYAN = (0, 242, 254)
ELECTRIC_BLUE = (79, 172, 254)
PURPLE = (155, 81, 224)
CORAL = (255, 65, 108)
EMERALD = (0, 245, 160)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (140, 160, 190)
TEXT_DIM = (70, 90, 120)

# Fonts
FONT_HERO = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 72)
FONT_TITLE_XL = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 60)
FONT_TITLE = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 44)
FONT_SUB = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 26)
FONT_BODY = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 22)
FONT_BADGE = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)
FONT_STAT_NUM = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 52)
FONT_MONO = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 20)
FONT_MONO_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf" if os.path.exists(r"C:\Windows\Fonts\consolab.ttf") else r"C:\Windows\Fonts\consola.ttf", 24)

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(val, max_val))

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def create_base_canvas(f):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Subtle cyber grid
    grid_spacing = 80
    for x in range(0, WIDTH, grid_spacing):
        draw.line([(x, 0), (x, HEIGHT)], fill=(12, 18, 32), width=1)
    for y in range(0, HEIGHT, grid_spacing):
        draw.line([(0, y), (WIDTH, y)], fill=(12, 18, 32), width=1)
        
    # Top Progress Bar
    draw.rectangle([0, 0, int(WIDTH * ((f + 1) / TOTAL_FRAMES)), 4], fill=CYAN)
    return img, draw

def draw_pill_badge(draw, text, x, y, bg_color=PURPLE, text_color=TEXT_WHITE):
    bbox = FONT_BADGE.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = 16, 6
    rect = [x, y, x + tw + px * 2, y + th + py * 2 + 2]
    draw.rounded_rectangle(rect, radius=12, fill=bg_color)
    draw.text((x + px, y + py - 2), text, font=FONT_BADGE, fill=text_color)
    return rect[2]

# --- ECG Signal Generator for Scene 1 ---
def ecg_waveform(t_cycle):
    """Generate realistic P-Q-R-S-T cardiac wave over cycle [0, 1]"""
    # Baseline
    val = 0.0
    # P wave around 0.2
    val += 0.15 * math.exp(-((t_cycle - 0.20) ** 2) / 0.0012)
    # Q wave around 0.35 (negative dip)
    val -= 0.18 * math.exp(-((t_cycle - 0.35) ** 2) / 0.0004)
    # R spike around 0.40 (tall peak)
    val += 1.00 * math.exp(-((t_cycle - 0.40) ** 2) / 0.00035)
    # S wave around 0.45 (sharp dip)
    val -= 0.32 * math.exp(-((t_cycle - 0.45) ** 2) / 0.00045)
    # T wave around 0.65 (medium bump)
    val += 0.25 * math.exp(-((t_cycle - 0.65) ** 2) / 0.0035)
    return val

# ==========================================
# SCENE 1: THE HOOK (Frames 0 - 119, 0-4s)
# ==========================================
def render_scene_1(f):
    img, draw = create_base_canvas(f)
    t = f / 120.0  # normalized time in scene [0, 1]
    
    # Fade in / out
    alpha = 1.0
    if f < 10:
        alpha = f / 10.0
    elif f > 110:
        alpha = (120 - f) / 10.0
    
    # Eyebrow Badge
    draw_pill_badge(draw, "QISKIT FALL FEST 2026 · PHASE 2", 120, 120, bg_color=(20, 45, 80), text_color=CYAN)
    draw_pill_badge(draw, "TRACK: QUANTUM MACHINE LEARNING", 435, 120, bg_color=(45, 25, 75), text_color=PURPLE)
    
    # Main Hook Title
    title_y = int(220 - 15 * (1 - ease_in_out(clamp(t * 3.0))))
    draw.text((120, title_y), "100,000 Heartbeats a Day.", font=FONT_HERO, fill=TEXT_WHITE)
    draw.text((120, title_y + 90), "Can Quantum Entanglement Detect Arrhythmia?", font=FONT_TITLE, fill=CYAN)
    draw.text((120, title_y + 160), "Transforming noisy 30-feature clinical ECG data into a 4-qubit Hilbert space.", font=FONT_SUB, fill=TEXT_MUTED)
    
    # Oscilloscope Card
    card_x, card_y, card_w, card_h = 120, 520, 1680, 440
    draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=16, fill=(14, 21, 38, 230), outline=(0, 242, 254, 100), width=2)
    
    # Oscilloscope Grid
    for gx in range(card_x + 60, card_x + card_w - 60, 120):
        draw.line([(gx, card_y + 30), (gx, card_y + card_h - 30)], fill=(22, 35, 60), width=1)
    for gy in range(card_y + 40, card_y + card_h - 40, 60):
        draw.line([(card_x + 30, gy), (card_x + card_w - 30, gy)], fill=(22, 35, 60), width=1)
        
    draw.text((card_x + 40, card_y + 30), "PATIENT ECG TELEMETRY · LEAD II (MIT-BIH DATABASE)", font=FONT_BADGE, fill=CYAN)
    draw.text((card_x + card_w - 360, card_y + 30), "REAL-TIME QUANTUM ENCODING", font=FONT_BADGE, fill=EMERALD)
    
    # Animated ECG Pulse line
    cy = card_y + 240
    num_points = 500
    sweep_progress = (f * 1.5) % card_w
    
    points = []
    for px in range(card_x + 40, card_x + card_w - 40, 4):
        rel_x = px - (card_x + 40)
        # ECG beat cycles every 400 pixels
        t_cycle = ((rel_x - sweep_progress * 0.8) % 400) / 400.0
        amp = ecg_waveform(t_cycle)
        y = cy - amp * 140
        points.append((px, y))
        
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        # Color based on amplitude (R-spike is red/coral, rest is cyan)
        color = CORAL if p1[1] < (cy - 70) else CYAN
        draw.line([p1, p2], fill=color, width=3)
        
    # Lead indicator pulse dot
    head_x = card_x + 40 + (int(sweep_progress) % (card_w - 80))
    t_head = ((head_x - sweep_progress * 0.8) % 400) / 400.0
    head_y = cy - ecg_waveform(t_head) * 140
    draw.ellipse([head_x - 8, head_y - 8, head_x + 8, head_y + 8], fill=TEXT_WHITE, outline=CYAN, width=3)
    
    # Stats badge at bottom of card
    draw.text((card_x + 40, card_y + card_h - 50), "30 Morphological & Statistical Features → 4 Principal Angles θ0..θ3", font=FONT_MONO, fill=TEXT_MUTED)
    
    return img

# ========================================================
# SCENE 2: QUANTUM ARCHITECTURE (Frames 120 - 239, 4-8s)
# ========================================================
def render_scene_2(f):
    img, draw = create_base_canvas(f)
    t = (f - 120) / 120.0
    
    # Eyebrow Badge
    draw_pill_badge(draw, "QUANTUM ARCHITECTURE · PENNYLANE & QISKIT", 120, 80, bg_color=(45, 25, 75), text_color=PURPLE)
    
    draw.text((120, 130), "4 Qubits. 24 Trainable Parameters.", font=FONT_TITLE_XL, fill=TEXT_WHITE)
    draw.text((120, 215), "Ry AngleEmbedding + StronglyEntanglingLayers (Circular CNOT Entanglement)", font=FONT_SUB, fill=CYAN)
    
    # Circuit Board Container
    cb_x, cb_y, cb_w, cb_h = 120, 290, 1100, 680
    draw.rounded_rectangle([cb_x, cb_y, cb_x + cb_w, cb_y + cb_h], radius=16, fill=(14, 21, 38, 230), outline=(0, 242, 254, 100), width=2)
    
    draw.text((cb_x + 40, cb_y + 35), "PARAMETERIZED VARIATIONAL QUANTUM CIRCUIT (VQC)", font=FONT_BADGE, fill=CYAN)
    draw.line([(cb_x + 40, cb_y + 75), (cb_x + cb_w - 40, cb_y + 75)], fill=(30, 45, 75), width=1)
    
    # 4 Quantum Wires
    wire_y_coords = [cb_y + 160, cb_y + 270, cb_y + 380, cb_y + 490]
    qubit_labels = ["|q0⟩", "|q1⟩", "|q2⟩", "|q3⟩"]
    
    for idx, wy in enumerate(wire_y_coords):
        # Qubit label
        draw.text((cb_x + 50, wy - 14), qubit_labels[idx], font=FONT_MONO_BOLD, fill=TEXT_WHITE)
        # Quantum wire
        draw.line([(cb_x + 130, wy), (cb_x + cb_w - 80, wy)], fill=(35, 60, 95), width=2)
        
    # Animated packet moving across circuits
    packet_x = cb_x + 150 + int((f * 8) % (cb_w - 300))
    for wy in wire_y_coords:
        draw.ellipse([packet_x - 6, wy - 6, packet_x + 6, wy + 6], fill=CYAN)
        
    # Stage 1: Feature Encoding Ry(θi)
    enc_x = cb_x + 180
    draw.text((enc_x + 10, cb_y + 100), "ENCODING", font=FONT_BADGE, fill=EMERALD)
    for idx, wy in enumerate(wire_y_coords):
        draw.rounded_rectangle([enc_x, wy - 30, enc_x + 85, wy + 30], radius=8, fill=(0, 120, 90), outline=EMERALD, width=2)
        draw.text((enc_x + 14, wy - 12), f"Ry(θ{idx})", font=FONT_MONO, fill=TEXT_WHITE)
        
    # Stage 2: Layer 1 Rot(φ,θ,ω) + CNOTs
    l1_x = cb_x + 350
    draw.text((l1_x + 60, cb_y + 100), "LAYER 1 (12 PARAMS)", font=FONT_BADGE, fill=PURPLE)
    for idx, wy in enumerate(wire_y_coords):
        draw.rounded_rectangle([l1_x, wy - 30, l1_x + 80, wy + 30], radius=8, fill=(70, 35, 110), outline=PURPLE, width=2)
        draw.text((l1_x + 18, wy - 12), "Rot", font=FONT_MONO, fill=TEXT_WHITE)
        
    # CNOT Ladder 1
    cnot1_x = cb_x + 470
    for q in range(4):
        target_q = (q + 1) % 4
        y_ctrl = wire_y_coords[q]
        y_tgt = wire_y_coords[target_q]
        draw.line([(cnot1_x + q * 25, y_ctrl), (cnot1_x + q * 25, y_tgt)], fill=CYAN, width=2)
        draw.ellipse([cnot1_x + q * 25 - 5, y_ctrl - 5, cnot1_x + q * 25 + 5, y_ctrl + 5], fill=CYAN)
        draw.ellipse([cnot1_x + q * 25 - 8, y_tgt - 8, cnot1_x + q * 25 + 8, y_tgt + 8], outline=CYAN, width=2)
        
    # Stage 3: Layer 2 Rot(φ,θ,ω) + CNOTs
    l2_x = cb_x + 630
    draw.text((l2_x + 60, cb_y + 100), "LAYER 2 (12 PARAMS)", font=FONT_BADGE, fill=PURPLE)
    for idx, wy in enumerate(wire_y_coords):
        draw.rounded_rectangle([l2_x, wy - 30, l2_x + 80, wy + 30], radius=8, fill=(70, 35, 110), outline=PURPLE, width=2)
        draw.text((l2_x + 18, wy - 12), "Rot", font=FONT_MONO, fill=TEXT_WHITE)
        
    # CNOT Ladder 2
    cnot2_x = cb_x + 750
    for q in range(4):
        target_q = (q + 1) % 4
        y_ctrl = wire_y_coords[q]
        y_tgt = wire_y_coords[target_q]
        draw.line([(cnot2_x + q * 25, y_ctrl), (cnot2_x + q * 25, y_tgt)], fill=CYAN, width=2)
        draw.ellipse([cnot2_x + q * 25 - 5, y_ctrl - 5, cnot2_x + q * 25 + 5, y_ctrl + 5], fill=CYAN)
        draw.ellipse([cnot2_x + q * 25 - 8, y_tgt - 8, cnot2_x + q * 25 + 8, y_tgt + 8], outline=CYAN, width=2)
        
    # Measurement stage
    meas_x = cb_x + 910
    draw.text((meas_x - 10, cb_y + 100), "MEASURE", font=FONT_BADGE, fill=CORAL)
    for wy in wire_y_coords:
        draw.rounded_rectangle([meas_x, wy - 25, meas_x + 60, wy + 25], radius=6, fill=(100, 30, 45), outline=CORAL, width=2)
        draw.text((meas_x + 18, wy - 12), "⟨Z⟩", font=FONT_MONO, fill=TEXT_WHITE)
        
    # Bottom callout
    draw.text((cb_x + 40, cb_y + cb_h - 60), "Circular CNOT topology natively mapped to Processor D 4-qubit ring (0 SWAP overhead).", font=FONT_MONO, fill=CYAN)
    
    # Right Column: Architectural Deep Dive Cards
    right_x, right_w = 1260, 540
    
    # Card 1: Dimensionality Reduction
    d1_y, d1_h = 290, 210
    draw.rounded_rectangle([right_x, d1_y, right_x + right_w, d1_y + d1_h], radius=16, fill=(14, 21, 38, 230), outline=(0, 242, 254, 80), width=2)
    draw.text((right_x + 35, d1_y + 25), "FEATURE SEPARATION (PCA)", font=FONT_BADGE, fill=CYAN)
    draw.text((right_x + 35, d1_y + 60), "81.3% Variance", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((right_x + 35, d1_y + 130), "PCA fitted strictly on 6,000 train samples.", font=FONT_BODY, fill=TEXT_MUTED)
    draw.text((right_x + 35, d1_y + 160), "0 data leakage into validation or test.", font=FONT_BODY, fill=EMERALD)
    
    # Card 2: Parameterization
    d2_y, d2_h = 525, 210
    draw.rounded_rectangle([right_x, d2_y, right_x + right_w, d2_y + d2_h], radius=16, fill=(14, 21, 38, 230), outline=(155, 81, 224, 80), width=2)
    draw.text((right_x + 35, d2_y + 25), "EXPRESSIBILITY & ANGLE SCALE", font=FONT_BADGE, fill=PURPLE)
    draw.text((right_x + 35, d2_y + 60), "24 Trainable Angles", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((right_x + 35, d2_y + 130), "2 layers of 4 x Rot(φ,θ,ω) Euler gates.", font=FONT_BODY, fill=TEXT_MUTED)
    draw.text((right_x + 35, d2_y + 160), "Adam optimizer (lr=0.08, batch=64).", font=FONT_BODY, fill=ELECTRIC_BLUE)
    
    # Card 3: Ablation Findings
    d3_y, d3_h = 760, 210
    draw.rounded_rectangle([right_x, d3_y, right_x + right_w, d3_y + d3_h], radius=16, fill=(14, 21, 38, 230), outline=(0, 245, 160, 80), width=2)
    draw.text((right_x + 35, d3_y + 25), "RIGOROUS ABLATION STUDY", font=FONT_BADGE, fill=EMERALD)
    draw.text((right_x + 35, d3_y + 60), "Sweet Spot: 2 Layers", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((right_x + 35, d3_y + 130), "1-Layer: 0.5447 (underfit) | 2-Layer: 0.9958", font=FONT_BODY, fill=TEXT_MUTED)
    draw.text((right_x + 35, d3_y + 160), "3-Layer: 1.0000 (3x longer runtime).", font=FONT_BODY, fill=TEXT_MUTED)
    
    return img

# ========================================================
# SCENE 3: THE EMPIRICAL VERDICT (Frames 240 - 359, 8-12s)
# ========================================================
def render_scene_3(f):
    img, draw = create_base_canvas(f)
    t = (f - 240) / 120.0
    
    # Top Header
    draw_pill_badge(draw, "CLINICAL EVALUATION · MIT-BIH VAL SET (1,500 BEATS)", 120, 80, bg_color=(25, 40, 70), text_color=CYAN)
    
    draw.text((120, 130), "Zero False Positives.", font=FONT_HERO, fill=TEXT_WHITE)
    draw.text((120, 225), "100% Precision on Normal Sinus Beats (899 / 899 Correct) · 0.9979 Macro-F1", font=FONT_SUB, fill=(160, 185, 215))
    
    # Left Card: Confusion Matrix Showcase
    cm_x, cm_y, cm_w, cm_h = 120, 290, 800, 680
    draw.rounded_rectangle([cm_x, cm_y, cm_x + cm_w, cm_y + cm_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 242, 254, 120), width=2)
    
    draw.text((cm_x + 40, cm_y + 35), "VQC-4Q CONFUSION MATRIX (IDEAL SIMULATION)", font=FONT_BADGE, fill=CYAN)
    draw.text((cm_x + 40, cm_y + 65), "Validated against ground truth MIT-BIH physician annotations", font=FONT_BODY, fill=TEXT_MUTED)
    draw.line([(cm_x + 40, cm_y + 105), (cm_x + cm_w - 40, cm_y + 105)], fill=(30, 45, 75), width=1)
    
    # Confusion Matrix Grid
    grid_ox, grid_oy = cm_x + 120, cm_y + 150
    cell_w, cell_h = 260, 180
    
    draw.text((grid_ox + 70, grid_oy - 40), "PREDICTED NORMAL", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((grid_ox + cell_w + 50, grid_oy - 40), "PREDICTED ARRHYTHMIA", font=FONT_BADGE, fill=TEXT_MUTED)
    
    # Cell 1: True Normal (899)
    draw.rounded_rectangle([grid_ox, grid_oy, grid_ox + cell_w, grid_oy + cell_h], radius=12, fill=(0, 120, 90, 80), outline=(0, 245, 160, 200), width=2)
    draw.text((grid_ox + 30, grid_oy + 30), "TRUE NORMAL", font=FONT_BADGE, fill=EMERALD)
    draw.text((grid_ox + 30, grid_oy + 65), "899", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((grid_ox + 30, grid_oy + 135), "100.0% Specificity", font=FONT_BODY, fill=EMERALD)
    
    # Cell 2: False Arrhythmia (0) -> Hero Stat!
    draw.rounded_rectangle([grid_ox + cell_w + 20, grid_oy, grid_ox + cell_w * 2 + 20, grid_oy + cell_h], radius=12, fill=(30, 20, 25, 80), outline=(50, 60, 90, 150), width=1)
    draw.text((grid_ox + cell_w + 50, grid_oy + 30), "FALSE POSITIVE (FP)", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((grid_ox + cell_w + 50, grid_oy + 65), "0", font=FONT_STAT_NUM, fill=(100, 255, 180))
    draw.text((grid_ox + cell_w + 50, grid_oy + 135), "0.00% Error Rate", font=FONT_BODY, fill=TEXT_MUTED)
    
    # Cell 3: False Normal (3)
    draw.rounded_rectangle([grid_ox, grid_oy + cell_h + 20, grid_ox + cell_w, grid_oy + cell_h * 2 + 20], radius=12, fill=(30, 20, 25, 80), outline=(50, 60, 90, 150), width=1)
    draw.text((grid_ox + 30, grid_oy + cell_h + 50), "FALSE NEGATIVE (FN)", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((grid_ox + 30, grid_oy + cell_h + 85), "3", font=FONT_STAT_NUM, fill=CORAL)
    draw.text((grid_ox + 30, grid_oy + cell_h + 155), "0.50% Miss Rate", font=FONT_BODY, fill=TEXT_MUTED)
    
    # Cell 4: True Arrhythmia (598)
    draw.rounded_rectangle([grid_ox + cell_w + 20, grid_oy + cell_h + 20, grid_ox + cell_w * 2 + 20, grid_oy + cell_h * 2 + 20], radius=12, fill=(0, 100, 150, 80), outline=(0, 242, 254, 200), width=2)
    draw.text((grid_ox + cell_w + 50, grid_oy + cell_h + 50), "TRUE ARRHYTHMIA", font=FONT_BADGE, fill=CYAN)
    draw.text((grid_ox + cell_w + 50, grid_oy + cell_h + 85), "598", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((grid_ox + cell_w + 50, grid_oy + cell_h + 155), "99.50% Sensitivity", font=FONT_BODY, fill=CYAN)
    
    draw.text((cm_x + 40, cm_y + cm_h - 60), "1,497 / 1,500 total predictions correct on holdout validation set.", font=FONT_MONO, fill=CYAN)
    
    # Right Side: HUD Cards
    right_x, right_w = 980, 820
    
    # Card 1: Macro-F1
    c1_y, c1_h = 290, 195
    draw.rounded_rectangle([right_x, c1_y, right_x + right_w, c1_y + c1_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 242, 254, 100), width=2)
    draw.text((right_x + 40, c1_y + 30), "PRIMARY COMPETITION METRIC", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((right_x + 40, c1_y + 65), "0.9979", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 68), fill=CYAN)
    draw.text((right_x + 360, c1_y + 85), "MACRO-F1 SCORE", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((right_x + 40, c1_y + 150), "Exceeds required Phase 2 competition threshold (F1 > 0.85)", font=FONT_BODY, fill=EMERALD)
    
    # Card 2: Accuracy & Speed
    c2_y, c2_h = 515, 195
    draw.rounded_rectangle([right_x, c2_y, right_x + right_w, c2_y + c2_h], radius=16, fill=(14, 21, 38, 240), outline=(155, 81, 224, 100), width=2)
    draw.text((right_x + 40, c2_y + 30), "OVERALL ACCURACY & RUNTIME", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((right_x + 40, c2_y + 65), "99.80%", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 68), fill=PURPLE)
    draw.text((right_x + 360, c2_y + 85), "40.08 s TRAINING", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((right_x + 40, c2_y + 150), "Vectorized Autograd batching: 50x faster than sequential simulation", font=FONT_BODY, fill=ELECTRIC_BLUE)
    
    # Card 3: Architecture Highlights
    c3_y, c3_h = 740, 230
    draw.rounded_rectangle([right_x, c3_y, right_x + right_w, c3_y + c3_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 245, 160, 100), width=2)
    draw.text((right_x + 40, c3_y + 25), "QUANTUM ARCHITECTURE HIGHLIGHTS", font=FONT_BADGE, fill=EMERALD)
    
    bullets = [
        ("4 Qubits · 2 Layers", "24 Trainable Euler rotation parameters Rot(φ,θ,ω)"),
        ("Y-axis AngleEmbedding", "PCA fitted strictly on train: 81.3% variance preserved"),
        ("Hardware-Matched Ansatz", "Circular CNOT topology matching Processor D native ring")
    ]
    for idx, (b_title, b_desc) in enumerate(bullets):
        by = c3_y + 65 + idx * 52
        draw.text((right_x + 40, by), f"• {b_title}:", font=FONT_MONO_BOLD, fill=TEXT_WHITE)
        draw.text((right_x + 340, by + 2), b_desc, font=FONT_BODY, fill=TEXT_MUTED)
        
    return img

# ========================================================
# SCENE 4: NOISE RESILIENCE & BENCHMARKS (Frames 360-479)
# ========================================================
def render_scene_4(f):
    img, draw = create_base_canvas(f)
    t = (f - 360) / 120.0
    
    # Eyebrow Badge
    draw_pill_badge(draw, "NISQ NOISE ANALYSIS & PROCESSOR BENCHMARKS", 120, 80, bg_color=(80, 25, 45), text_color=CORAL)
    
    draw.text((120, 130), "Tested Under Heavy Depolarizing Noise.", font=FONT_TITLE_XL, fill=TEXT_WHITE)
    draw.text((120, 215), "Evaluated under official QFF2026-v2 noise model (p1=0.01, p2=0.05, readout=0.02)", font=FONT_SUB, fill=CYAN)
    
    # Left Card: Noise Battle on PS-Q1
    n_x, n_y, n_w, n_h = 120, 290, 800, 680
    draw.rounded_rectangle([n_x, n_y, n_x + n_w, n_y + n_h], radius=16, fill=(14, 21, 38, 240), outline=(255, 65, 108, 120), width=2)
    
    draw.text((n_x + 40, n_y + 35), "PS-Q1 NOISE RESILIENCE (1,500 CIRCUITS ON AERSIMULATOR)", font=FONT_BADGE, fill=CORAL)
    draw.text((n_x + 40, n_y + 65), "Each validation circuit transpiled to [rx, ry, rz, cx] and sampled at 8,192 shots", font=FONT_BODY, fill=TEXT_MUTED)
    draw.line([(n_x + 40, n_y + 105), (n_x + n_w - 40, n_y + 105)], fill=(30, 45, 75), width=1)
    
    # Noise parameter pills
    draw_pill_badge(draw, "1-Qubit Gate Error: 1%", n_x + 40, n_y + 125, bg_color=(25, 35, 60), text_color=TEXT_WHITE)
    draw_pill_badge(draw, "2-Qubit CX Error: 5%", n_x + 280, n_y + 125, bg_color=(45, 25, 40), text_color=CORAL)
    draw_pill_badge(draw, "Readout Error: 2%", n_x + 520, n_y + 125, bg_color=(25, 45, 50), text_color=CYAN)
    
    # Comparison Bars
    bar_ox = n_x + 40
    
    # Ideal Metric
    draw.text((bar_ox, n_y + 200), "IDEAL SIMULATION (MACRO-F1)", font=FONT_BADGE, fill=CYAN)
    draw.text((bar_ox + 600, n_y + 195), "0.9979", font=FONT_TITLE, fill=CYAN)
    draw.rounded_rectangle([bar_ox, n_y + 250, bar_ox + 720, n_y + 285], radius=8, fill=(20, 30, 50))
    draw.rounded_rectangle([bar_ox, n_y + 250, bar_ox + int(720 * 0.9979), n_y + 285], radius=8, fill=CYAN)
    
    # Noisy Metric
    draw.text((bar_ox, n_y + 320), "NOISY QISKIT AER SIMULATION (MACRO-F1)", font=FONT_BADGE, fill=CORAL)
    draw.text((bar_ox + 600, n_y + 315), "0.9965", font=FONT_TITLE, fill=CORAL)
    draw.rounded_rectangle([bar_ox, n_y + 370, bar_ox + 720, n_y + 405], radius=8, fill=(20, 30, 50))
    draw.rounded_rectangle([bar_ox, n_y + 370, bar_ox + int(720 * 0.9965), n_y + 405], radius=8, fill=CORAL)
    
    # Delta Callout Box
    delta_y = n_y + 450
    draw.rounded_rectangle([bar_ox, delta_y, bar_ox + 720, delta_y + 160], radius=12, fill=(0, 60, 45, 90), outline=EMERALD, width=2)
    draw.text((bar_ox + 30, delta_y + 25), "EMPIRICAL RESILIENCE VERDICT", font=FONT_BADGE, fill=EMERALD)
    draw.text((bar_ox + 30, delta_y + 55), "Δ Macro-F1: only -0.0014 drop (-0.14%)", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((bar_ox + 30, delta_y + 115), "Still 0 False Positives under noise (899 / 899 Normal beats classified correctly).", font=FONT_BODY, fill=CYAN)
    
    # Right Column: Task A & Task B Badges
    r_x, r_w = 980, 820
    
    # Task A Card: Bell State Fidelity
    t1_y, t1_h = 290, 320
    draw.rounded_rectangle([r_x, t1_y, r_x + r_w, t1_y + t1_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 242, 254, 100), width=2)
    draw_pill_badge(draw, "TASK A · OFFICIAL BENCHMARK", r_x + 40, t1_y + 30, bg_color=(20, 50, 80), text_color=CYAN)
    draw.text((r_x + 40, t1_y + 75), "Bell-State Fidelity (|Φ+⟩)", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((r_x + 40, t1_y + 130), "F_ideal = 1.0000 ± 0.0000", font=FONT_SUB, fill=CYAN)
    draw.text((r_x + 40, t1_y + 170), "F_noisy = 0.8801 ± 0.0026", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 50), fill=EMERALD)
    draw.text((r_x + 40, t1_y + 245), "Exact match to official challenge kit reference (F ≈ 0.880).", font=FONT_BODY, fill=TEXT_MUTED)
    draw.text((r_x + 40, t1_y + 275), "Measured using XX, YY, ZZ parities over 8,192 shots.", font=FONT_MONO, fill=TEXT_MUTED)
    
    # Task B Card: Processor D
    t2_y, t2_h = 650, 320
    draw.rounded_rectangle([r_x, t2_y, r_x + r_w, t2_y + t2_h], radius=16, fill=(14, 21, 38, 240), outline=(155, 81, 224, 100), width=2)
    draw_pill_badge(draw, "TASK B · OFFICIAL BENCHMARK", r_x + 40, t2_y + 30, bg_color=(45, 25, 75), text_color=PURPLE)
    draw.text((r_x + 40, t2_y + 75), "Processor D: 4-Qubit Ring Topology", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((r_x + 40, t2_y + 130), "QFT-3 Benchmark: F_ideal = 1.000000", font=FONT_SUB, fill=CYAN)
    draw.text((r_x + 40, t2_y + 170), "F_noisy = 0.626784", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 50), fill=PURPLE)
    draw.text((r_x + 40, t2_y + 245), "Transpiled: 21 Depth · 12 CX Gates · Only 1 SWAP Overhead.", font=FONT_BODY, fill=TEXT_WHITE)
    draw.text((r_x + 40, t2_y + 275), "100% compliant with official schema & validated by benchmark CLI.", font=FONT_MONO, fill=EMERALD)
    
    return img

# ========================================================
# SCENE 5: OUTRO & VICTORY (Frames 480 - 599, 16-20s)
# ========================================================
def render_scene_5(f):
    img, draw = create_base_canvas(f)
    t = (f - 480) / 120.0
    
    # Center Hero Lockup
    center_y = 120
    
    draw_pill_badge(draw, "COMPETITION SUBMISSION READY · PHASE 2", WIDTH//2 - 210, center_y, bg_color=(20, 50, 80), text_color=CYAN)
    
    # Team Logo
    team_name = "NEXUS-OPS"
    t_bbox = FONT_HERO.getbbox(team_name)
    tw = t_bbox[2] - t_bbox[0]
    draw.text((WIDTH//2 - tw//2, center_y + 55), team_name, font=FONT_HERO, fill=TEXT_WHITE)
    
    sub_title = "PS-Q1: Biomedical Signal Classification · Qiskit Fall Fest 2026"
    st_bbox = FONT_SUB.getbbox(sub_title)
    stw = st_bbox[2] - st_bbox[0]
    draw.text((WIDTH//2 - stw//2, center_y + 155), sub_title, font=FONT_SUB, fill=CYAN)
    
    # 4 Checklist Cards
    grid_ox, grid_oy = 240, 380
    c_w, c_h = 690, 240
    
    cards = [
        ("AUTOMATED TEST SUITE", "6 / 6 Tests Passing", "Dataset, classical baseline, VQC Autograd, Task A Bell state, Task B Processor D, Predictions format all green.", EMERALD),
        ("OFFICIAL PREDICTIONS", "predictions_q1.csv Validated", "1,500 test predictions verified with tools/validate_predictions.py q1. Exactly matching required schema.", CYAN),
        ("REPRODUCIBLE NOTEBOOK", "solution.ipynb Executed", "Ran top-to-bottom with jupyter nbconvert --execute (0 errors, full execution outputs preserved).", PURPLE),
        ("COMPREHENSIVE REPORT", "reports/results.md Complete", "Full executive summary, quantum architecture rationale, noise comparison tables, and Processor D trade-offs.", ELECTRIC_BLUE)
    ]
    
    positions = [
        (grid_ox, grid_oy),
        (grid_ox + c_w + 60, grid_oy),
        (grid_ox, grid_oy + c_h + 40),
        (grid_ox + c_w + 60, grid_oy + c_h + 40)
    ]
    
    for idx, (head, status, desc, col) in enumerate(cards):
        gx, gy = positions[idx]
        draw.rounded_rectangle([gx, gy, gx + c_w, gy + c_h], radius=16, fill=(14, 21, 38, 240), outline=(col[0], col[1], col[2], 120), width=2)
        draw.text((gx + 35, gy + 30), head, font=FONT_BADGE, fill=TEXT_MUTED)
        draw.text((gx + 35, gy + 65), f"✔ {status}", font=FONT_TITLE, fill=col)
        draw.text((gx + 35, gy + 140), desc, font=FONT_BODY, fill=TEXT_WHITE)
        
    # Footer Callout
    footer_text = "NEXUS-OPS · Quantum Solutions for Clinical Precision · October 2026"
    f_bbox = FONT_MONO.getbbox(footer_text)
    fw = f_bbox[2] - f_bbox[0]
    draw.text((WIDTH//2 - fw//2, HEIGHT - 70), footer_text, font=FONT_MONO, fill=TEXT_MUTED)
    
    return img

def main():
    frames_dir = r"C:\QFF\QFF2026-PHASE2\NEXUS_OPS_PS_Q1\brag-output\work\frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Rendering {TOTAL_FRAMES} frames (1920x1080 @ 30fps) to {frames_dir}...")
    
    for f in range(TOTAL_FRAMES):
        if f < 120:
            frame_img = render_scene_1(f)
        elif f < 240:
            frame_img = render_scene_2(f)
        elif f < 360:
            frame_img = render_scene_3(f)
        elif f < 480:
            frame_img = render_scene_4(f)
        else:
            frame_img = render_scene_5(f)
            
        frame_path = os.path.join(frames_dir, f"frame_{f:04d}.png")
        frame_img.save(frame_path)
        
        if (f + 1) % 60 == 0:
            print(f"Rendered {f + 1}/{TOTAL_FRAMES} frames ({(f+1)/30:.1f}s)...")
            
    print("All 600 frames rendered successfully!")
    
    # Save settled poster frame from Scene 3 (frame 330)
    poster_path = r"C:\QFF\QFF2026-PHASE2\NEXUS_OPS_PS_Q1\brag-output\brag.jpg"
    poster_img = render_scene_3(330)
    poster_img.save(poster_path, quality=95)
    print(f"Saved hero poster thumbnail to {poster_path}")

if __name__ == "__main__":
    main()
