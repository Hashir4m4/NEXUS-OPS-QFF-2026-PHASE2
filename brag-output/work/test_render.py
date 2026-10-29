"""
NEXUS-OPS PS-Q1 Launch Video Renderer
High-fidelity frame generation using PIL + NumPy
Resolution: 1920x1080 @ 30fps, 20 seconds = 600 frames
"""
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH, HEIGHT = 1920, 1080
FPS = 30
TOTAL_FRAMES = 600

# Color Palette
BG_COLOR = (7, 10, 20)           # #070a14
NAVY_CARD = (14, 21, 38)         # #0e1526
CARD_BORDER = (0, 242, 254, 70)   # Cyan border with alpha
CYAN = (0, 242, 254)             # #00f2fe
ELECTRIC_BLUE = (79, 172, 254)   # #4facfe
PURPLE = (155, 81, 224)          # #9b51e0
CORAL = (255, 65, 108)           # #ff416c
EMERALD = (0, 245, 160)          # #00f5a0
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (140, 160, 190)
TEXT_DIM = (80, 100, 130)

# Fonts
FONT_TITLE_XL = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 64)
FONT_TITLE = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 46)
FONT_SUB = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 26)
FONT_BODY = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 22)
FONT_BADGE = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 18)
FONT_STAT_NUM = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 54)
FONT_MONO = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 20)
FONT_MONO_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\consolab.ttf" if os.path.exists(r"C:\Windows\Fonts\consolab.ttf") else r"C:\Windows\Fonts\consola.ttf", 24)

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(val, max_val))

def create_base_canvas():
    """Create a dark cinematic background with subtle grid lines and radial vignette."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Subtle grid lines
    grid_spacing = 80
    for x in range(0, WIDTH, grid_spacing):
        draw.line([(x, 0), (x, HEIGHT)], fill=(12, 18, 32), width=1)
    for y in range(0, HEIGHT, grid_spacing):
        draw.line([(0, y), (WIDTH, y)], fill=(12, 18, 32), width=1)
        
    return img

def draw_pill_badge(draw, text, x, y, bg_color=PURPLE, text_color=TEXT_WHITE):
    bbox = FONT_BADGE.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = 16, 6
    rect = [x, y, x + tw + px * 2, y + th + py * 2 + 2]
    draw.rounded_rectangle(rect, radius=12, fill=bg_color)
    draw.text((x + px, y + py - 2), text, font=FONT_BADGE, fill=text_color)
    return rect[2]  # return right edge

def render_scene_3(f, t_norm):
    """
    Scene 3: The Empirical Verdict (Frames 240-374)
    Hero frame showcasing the 0 False Positives confusion matrix and 0.9979 Macro-F1.
    """
    img = create_base_canvas()
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Progress Bar at very top
    draw.rectangle([0, 0, int(WIDTH * (f / TOTAL_FRAMES)), 4], fill=CYAN)
    
    # Top Header
    draw_pill_badge(draw, "CLINICAL EVALUATION · MIT-BIH VAL SET (1,500 BEATS)", 120, 80, bg_color=(25, 40, 70), text_color=CYAN)
    
    title_alpha = int(255 * clamp(t_norm * 4))
    draw.text((120, 130), "Zero False Positives.", font=FONT_TITLE_XL, fill=(255, 255, 255, title_alpha))
    draw.text((120, 215), "100% Precision on Normal Sinus Beats (899 / 899 Correct) · 0.9979 Macro-F1", font=FONT_SUB, fill=(160, 185, 215, title_alpha))
    
    # Left Card: Confusion Matrix Showcase
    cm_x, cm_y, cm_w, cm_h = 120, 290, 800, 680
    draw.rounded_rectangle([cm_x, cm_y, cm_x + cm_w, cm_y + cm_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 242, 254, 120), width=2)
    
    # Card Header
    draw.text((cm_x + 40, cm_y + 35), "VQC-4Q CONFUSION MATRIX (IDEAL SIMULATION)", font=FONT_BADGE, fill=CYAN)
    draw.text((cm_x + 40, cm_y + 65), "Validated against ground truth MIT-BIH physician annotations", font=FONT_BODY, fill=TEXT_MUTED)
    draw.line([(cm_x + 40, cm_y + 105), (cm_x + cm_w - 40, cm_y + 105)], fill=(30, 45, 75), width=1)
    
    # Confusion Matrix Grid
    grid_ox, grid_oy = cm_x + 120, cm_y + 150
    cell_w, cell_h = 260, 180
    
    # Labels
    draw.text((grid_ox + 70, grid_oy - 40), "PREDICTED NORMAL", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((grid_ox + cell_w + 50, grid_oy - 40), "PREDICTED ARRHYTHMIA", font=FONT_BADGE, fill=TEXT_MUTED)
    
    # Cell 1: True Normal (899)
    draw.rounded_rectangle([grid_ox, grid_oy, grid_ox + cell_w, grid_oy + cell_h], radius=12, fill=(0, 120, 90, 80), outline=(0, 245, 160, 200), width=2)
    draw.text((grid_ox + 30, grid_oy + 30), "TRUE NORMAL", font=FONT_BADGE, fill=EMERALD)
    draw.text((grid_ox + 30, grid_oy + 65), "899", font=FONT_STAT_NUM, fill=TEXT_WHITE)
    draw.text((grid_ox + 30, grid_oy + 135), "100.0% Specificity", font=FONT_BODY, fill=EMERALD)
    
    # Cell 2: False Arrhythmia (0) -> Highlighted Hero Stat!
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
    
    # Bottom callout in Card
    draw.text((cm_x + 40, cm_y + cm_h - 60), "1,497 / 1,500 total predictions correct on holdout validation set.", font=FONT_MONO, fill=CYAN)
    
    # Right Side: 3 Big Metric HUD Cards
    right_x, right_w = 980, 820
    
    # HUD Card 1: Macro-F1
    c1_y, c1_h = 290, 195
    draw.rounded_rectangle([right_x, c1_y, right_x + right_w, c1_y + c1_h], radius=16, fill=(14, 21, 38, 240), outline=(0, 242, 254, 100), width=2)
    draw.text((right_x + 40, c1_y + 30), "PRIMARY COMPETITION METRIC", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((right_x + 40, c1_y + 65), "0.9979", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 68), fill=CYAN)
    draw.text((right_x + 360, c1_y + 85), "MACRO-F1 SCORE", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((right_x + 40, c1_y + 150), "Exceeds required Phase 2 competition threshold (F1 > 0.85)", font=FONT_BODY, fill=EMERALD)
    
    # HUD Card 2: Accuracy & Speed
    c2_y, c2_h = 515, 195
    draw.rounded_rectangle([right_x, c2_y, right_x + right_w, c2_y + c2_h], radius=16, fill=(14, 21, 38, 240), outline=(155, 81, 224, 100), width=2)
    draw.text((right_x + 40, c2_y + 30), "OVERALL ACCURACY & RUNTIME", font=FONT_BADGE, fill=TEXT_MUTED)
    draw.text((right_x + 40, c2_y + 65), "99.80%", font=ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 68), fill=PURPLE)
    draw.text((right_x + 360, c2_y + 85), "40.08 s TRAINING", font=FONT_TITLE, fill=TEXT_WHITE)
    draw.text((right_x + 40, c2_y + 150), "Vectorized Autograd batching: 50x faster than sequential simulation", font=FONT_BODY, fill=ELECTRIC_BLUE)
    
    # HUD Card 3: Model Architecture summary
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

if __name__ == "__main__":
    print("Testing Scene 3 render...")
    test_img = render_scene_3(330, 0.8)
    out_path = r"C:\QFF\QFF2026-PHASE2\NEXUS_OPS_PS_Q1\brag-output\work\test_frame.png"
    test_img.save(out_path)
    print(f"Saved test frame to {out_path}")
