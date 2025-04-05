import os
import re
import cv2
import json
import time
import random
import asyncio
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from twitchio.ext import commands
from transformers import CLIPProcessor, CLIPModel
import colorsys
from datetime import datetime
from collections import Counter

# === CONFIG ===
TOKEN = 'jyjt32hzstigunilk74p33g351j2vb'
NICK = 'Honey'
CHANNEL = 'renarta_caylen'
CLIENT_ID = 'gp762nuuoqcoxypju8c569th9wz7q5'
EMOJI_FILE = "learned_amigurumi/emoji_db.json"
AMIGURUMI_DATA_FILE = "learned_amigurumi/amigurumi_data.json"
AMIGURUMI_COUNT_FILE = "learned_amigurumi/amigurumi_count.json"
FALLBACK_VIDEO_PATH = "old_live_video.mp4"
SCREENSHOT_DIR = "learned_amigurumi"
STATS_FILE = "learned_amigurumi/stats.json"

# === MODEL ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Theme Config ===
THEMES = {
    "Pink Kawaii": {
        "bg": "#FFD1DC",
        "fg": "#C71585",
        "text": "#4B0082",
        "chat_bg": "#FFE4ED",
        "button": "#FF69B4",
        "button_text": "white",
        "border": "#F48FB1",
        "title": "#8B008B",
        "accent": "#FF1493",
        "highlight": "#FFC0CB"
    },
    "Cozy Dark": {
        "bg": "#35281E",
        "fg": "#D2B48C",
        "text": "#F5DEB3",
        "chat_bg": "#483C32",
        "button": "#8B4513",
        "button_text": "#F5F5DC",
        "border": "#A0522D",
        "title": "#DEB887",
        "accent": "#CD853F",
        "highlight": "#D2691E"
    },
    "Pastel Garden": {
        "bg": "#E6F5FF",
        "fg": "#9370DB",
        "text": "#6A5ACD",
        "chat_bg": "#F0FFF0",
        "button": "#98FB98",
        "button_text": "#4B0082",
        "border": "#CCCCFF",
        "title": "#6A5ACD",
        "accent": "#9ACD32",
        "highlight": "#E0FFFF"
    }
}

# === Emoji Functions ===
def load_emojis():
    if not os.path.exists(EMOJI_FILE): return []
    with open(EMOJI_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_emoji(new_emoji):
    emojis = load_emojis()
    if new_emoji not in emojis:
        emojis.append(new_emoji)
        with open(EMOJI_FILE, "w", encoding="utf-8") as f:
            json.dump(emojis, f, ensure_ascii=False)

def extract_emojis(text):
    emoji_regex = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u2B55]+", re.UNICODE)
    return emoji_regex.findall(text)

def add_random_learned_emoji(message):
    emojis = load_emojis()
    if not emojis: return message
    return message + " " + "".join(random.choices(emojis, k=random.randint(1, 2)))

def get_top_emojis(limit=3):
    emojis = load_emojis()
    if not emojis: return []
    emoji_counts = Counter(emojis)
    return emoji_counts.most_common(limit)

# === Stats Functions ===
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "sessions": 0,
        "last_session": "",
        "colors_used": {},
        "detection_count": 0
    }

def save_stats(stats_data):
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

def update_stats(color=None):
    stats = load_stats()
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Increment session if it's a new day
    if stats["last_session"] != current_date:
        stats["sessions"] += 1
        stats["last_session"] = current_date
    
    # Update color usage
    if color:
        stats["colors_used"][color] = stats["colors_used"].get(color, 0) + 1
    
    # Increment detection counter
    stats["detection_count"] += 1
    
    save_stats(stats)
    return stats

def get_top_amigurumi(limit=3):
    count_data = load_amigurumi_count()
    if not count_data: return []
    return sorted(count_data.items(), key=lambda x: x[1], reverse=True)[:limit]

def get_favorite_color():
    stats = load_stats()
    if not stats.get("colors_used"): return "None"
    return max(stats["colors_used"].items(), key=lambda x: x[1])[0]

# === Amigurumi Recognition ===
def load_amigurumi_data():
    if os.path.exists(AMIGURUMI_DATA_FILE):
        with open(AMIGURUMI_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_amigurumi_data(data):
    with open(AMIGURUMI_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_amigurumi_count():
    if os.path.exists(AMIGURUMI_COUNT_FILE):
        with open(AMIGURUMI_COUNT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_amigurumi_count(count_data):
    with open(AMIGURUMI_COUNT_FILE, 'w', encoding='utf-8') as f:
        json.dump(count_data, f, ensure_ascii=False, indent=2)

def increment_amigurumi_count(label):
    count_data = load_amigurumi_count()
    count_data[label] = count_data.get(label, 0) + 1
    save_amigurumi_count(count_data)
    return count_data[label]

def recognize_amigurumi(img):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = load_amigurumi_data()
    if not data:
        return "Unknown amigurumi", False
    labels = list(data.keys())
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_text.softmax(dim=0)
    best_idx = probs.argmax().item()
    label = labels[best_idx]
    if probs[best_idx] < 0.8:
        return "Unknown amigurumi", False
    return label, True

def learn_amigurumi(img, label):
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    filename = f"{SCREENSHOT_DIR}/{label.replace(' ', '_')}_{int(time.time())}.jpg"
    cv2.imwrite(filename, img)
    data = load_amigurumi_data()
    if label not in data:
        data[label] = []
    data[label].append(filename)
    save_amigurumi_data(data)
    increment_amigurumi_count(label)

# === Color and Progress ===
def detect_dominant_color(img):
    image = cv2.resize(img, (50, 50))
    data = image.reshape((-1, 3)).astype(np.float32)
    _, _, centers = cv2.kmeans(data, 1, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    b, g, r = centers[0]
    rgb = (int(r), int(g), int(b))
    color_name = get_closest_color_name(rgb)
    hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
    friendly_name = get_friendly_color_name(rgb)
    return color_name, hex_color, friendly_name

def get_closest_color_name(rgb):
    from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb
    min_colors = {}
    for hex_val, name in CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = hex_to_rgb(hex_val)
        rd = (r_c - rgb[0]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())].replace("_", " ").capitalize()

def get_friendly_color_name(rgb):
    r, g, b = rgb
    
    # Convert to HSL for better naming
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    
    # Define base colors with friendly prefixes
    if s < 0.1:
        if l < 0.2: return "Charcoal Black"
        if l < 0.4: return "Smoky Gray"
        if l < 0.6: return "Soft Gray"
        if l < 0.8: return "Pearl Gray"
        return "Snow White"
    
    # Basic color wheel with cute names
    hue = h * 360
    prefix = ""
    if l < 0.3: prefix = "Deep "
    elif l < 0.4: prefix = "Dark "
    elif l > 0.8: prefix = "Pastel "
    elif l > 0.7: prefix = "Light "
    elif l > 0.6: prefix = "Soft "
    
    if s < 0.4:
        prefix = "Dusty " 
    
    if hue < 20: return prefix + "Ruby Red"
    elif hue < 40: return prefix + "Tangerine"
    elif hue < 60: return prefix + "Honey Gold"
    elif hue < 80: return prefix + "Lime Green"
    elif hue < 160: return prefix + "Mint Green"
    elif hue < 180: return prefix + "Aqua"
    elif hue < 220: return prefix + "Sky Blue"
    elif hue < 240: return prefix + "Blueberry"
    elif hue < 280: return prefix + "Lavender"
    elif hue < 320: return prefix + "Bubblegum Pink"
    elif hue < 340: return prefix + "Strawberry Pink"
    else: return prefix + "Cherry Red"

def estimate_progress():
    return random.choice(["just started", "making progress", "halfway there", "almost done"])

# === Twitch Bot ===
class Bot(commands.Bot):
    def __init__(self):
        super().__init__(token=TOKEN, prefix="!", initial_channels=[CHANNEL])
        self.recent_messages = []
        self.is_live = True
        self.waiting_for_label = False

    async def event_ready(self):
        print(f"Logged in as | {self.nick}")

    async def event_message(self, message):
        emojis = extract_emojis(message.content)
        for emoji in emojis:
            save_emoji(emoji)
        await self.handle_commands(message)

    async def event_usernotice(self, message):
        if message.tags["msg-id"] == "sub":
            await message.channel.send("Thanks for the sub! 🧶💖")
        elif message.tags["msg-id"] == "raid":
            await message.channel.send("A raid! Welcome raiders! 🎉✨")

    async def run_bot_logic(self):
        while True:
            ret, frame = app.get_latest_frame()
            if not ret:
                await asyncio.sleep(5)
                continue

            if self.waiting_for_label:
                await asyncio.sleep(2)
                continue

            label, known = recognize_amigurumi(frame)
            if not known:
                self.waiting_for_label = True
                app.ask_user_to_label(frame)
                await asyncio.sleep(2)
                continue

            color_name, hex_color, friendly_name = detect_dominant_color(frame)
            progress = estimate_progress()
            count = increment_amigurumi_count(label)
            msg = f"You're crocheting {label} (#{count}), using {friendly_name} yarn and you're {progress}!"
            msg = add_random_learned_emoji(msg)

            # Update stats
            update_stats(color_name)

            if msg not in self.recent_messages:
                self.recent_messages.append(msg)
                if len(self.recent_messages) > 10:
                    self.recent_messages.pop(0)
                if self.is_live:
                    chan = self.get_channel(CHANNEL)
                    await chan.send(msg)
                else:
                    app.update_display(label, color_name, hex_color, friendly_name, progress, msg)

            await asyncio.sleep(60)

# === GUI ===
class FloatingEmoji:
    def __init__(self, canvas, emoji, x, y):
        self.canvas = canvas
        self.emoji = emoji
        self.id = canvas.create_text(x, y, text=emoji, font=('Comic Sans MS', 20))
        self.dx = random.uniform(-2, 2)
        self.dy = random.uniform(-3, -1)
        self.alpha = 1.0
        self.fade_speed = random.uniform(0.01, 0.03)
        
    def update(self):
        self.canvas.move(self.id, self.dx, self.dy)
        self.alpha -= self.fade_speed
        if self.alpha <= 0:
            self.canvas.delete(self.id)
            return False
        return True

class CrochetApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧶 Rena's Crochet Monitor (Pink Kawaii)")
        self.current_theme = "Pink Kawaii"
        self.apply_theme(self.current_theme)
        
        # Store all displays
        self.stored_frame = None
        self.latest_frame = None
        self.floating_emojis = []
        self.canvas_emojis = None
        
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tab frames
        self.tab_live = tk.Frame(self.notebook, bg=THEMES[self.current_theme]["bg"])
        self.tab_stats = tk.Frame(self.notebook, bg=THEMES[self.current_theme]["bg"])
        self.tab_gallery = tk.Frame(self.notebook, bg=THEMES[self.current_theme]["bg"])
        self.tab_settings = tk.Frame(self.notebook, bg=THEMES[self.current_theme]["bg"])
        
        # Add tabs to notebook with emojis
        self.notebook.add(self.tab_live, text="📺 Live View")
        self.notebook.add(self.tab_stats, text="📊 Stats")
        self.notebook.add(self.tab_gallery, text="🧸 Gallery")
        self.notebook.add(self.tab_settings, text="🎀 Settings")
        
        # Setup each tab content
        self.setup_live_tab()
        self.setup_stats_tab()
        self.setup_gallery_tab()
        self.setup_settings_tab()
        
        # Create a hidden labeling dialog (will be shown when needed)
        self.setup_label_dialog()
        
        # Initialize webcam
        self.cap = self.init_video_capture()

    def apply_theme(self, theme_name):
        style = ttk.Style()
        style.theme_use('default')
        
        theme = THEMES[theme_name]
        
        # Configure styles
        style.configure('TNotebook', background=theme["bg"])
        style.configure('TNotebook.Tab', background=theme["highlight"], 
                       foreground=theme["text"], padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', theme["bg"])])
        
        # Apply to root
        self.root.configure(bg=theme["bg"])
        self.current_theme = theme_name

    def setup_live_tab(self):
        # Top frame for webcam and info
        top_frame = tk.Frame(self.tab_live, bg=THEMES[self.current_theme]["bg"])
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Webcam with decorative frame
        webcam_frame = tk.Frame(top_frame, bg=THEMES[self.current_theme]["border"], 
                              bd=5, relief=tk.GROOVE)
        webcam_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(webcam_frame, bg=THEMES[self.current_theme]["bg"], bd=0)
        self.video_label.pack(padx=2, pady=2)
        
        # Right side - Info Display
        info_frame = tk.Frame(top_frame, bg=THEMES[self.current_theme]["bg"])
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Amigurumi info with larger font
        self.label_text = tk.StringVar(value="Detected: -")
        self.label_display = tk.Label(info_frame, textvariable=self.label_text,
                                    font=('Comic Sans MS', 16, 'bold'), 
                                    fg=THEMES[self.current_theme]["fg"], 
                                    bg=THEMES[self.current_theme]["bg"])
        self.label_display.pack(anchor='w', pady=5)
        
        # Color info with swatch
        color_frame = tk.Frame(info_frame, bg=THEMES[self.current_theme]["bg"])
        color_frame.pack(fill=tk.X, pady=5)
        
        self.color_text = tk.StringVar(value="Yarn Color: -")
        self.color_display = tk.Label(color_frame, textvariable=self.color_text,
                                    font=('Comic Sans MS', 14), 
                                    fg=THEMES[self.current_theme]["fg"], 
                                    bg=THEMES[self.current_theme]["bg"])
        self.color_display.pack(side=tk.LEFT)
        
        self.color_swatch = tk.Canvas(color_frame, width=25, height=25, 
                                     bg=THEMES[self.current_theme]["bg"], 
                                     highlightthickness=0)
        self.color_swatch.pack(side=tk.LEFT, padx=10)
        self.swatch_rect = self.color_swatch.create_rectangle(0, 0, 25, 25, fill="#FFFFFF", outline="#000000")
        
        self.friendly_color_text = tk.StringVar(value="")
        self.friendly_color_display = tk.Label(info_frame, textvariable=self.friendly_color_text,
                                             font=('Comic Sans MS', 12, 'italic'), 
                                             fg=THEMES[self.current_theme]["fg"], 
                                             bg=THEMES[self.current_theme]["bg"])
        self.friendly_color_display.pack(anchor='w', padx=25)
        
        # Progress info
        self.progress_text = tk.StringVar(value="Progress: -")
        self.progress_display = tk.Label(info_frame, textvariable=self.progress_text,
                                       font=('Comic Sans MS', 14), 
                                       fg=THEMES[self.current_theme]["fg"], 
                                       bg=THEMES[self.current_theme]["bg"])
        self.progress_display.pack(anchor='w', pady=5)
        
        # Bottom frame for chat
        bottom_frame = tk.Frame(self.tab_live, bg=THEMES[self.current_theme]["bg"])
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Chat header with emoji canvas
        chat_header_frame = tk.Frame(bottom_frame, bg=THEMES[self.current_theme]["bg"])
        chat_header_frame.pack(fill=tk.X)
        
        self.chat_box_label = tk.Label(chat_header_frame, text="💬 Crochet Chat", 
                                      font=('Comic Sans MS', 14, 'bold'),
                                      bg=THEMES[self.current_theme]["bg"], 
                                      fg=THEMES[self.current_theme]["title"])
        self.chat_box_label.pack(side=tk.LEFT, pady=(10, 0))
        
        # Canvas for floating emojis
        self.canvas_emojis = tk.Canvas(chat_header_frame, width=150, height=30, 
                                      bg=THEMES[self.current_theme]["bg"], 
                                      highlightthickness=0)
        self.canvas_emojis.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Chat box
        self.chat_box = tk.Text(bottom_frame, height=8, width=60, font=('Comic Sans MS', 12),
                              bg=THEMES[self.current_theme]["chat_bg"], 
                              fg=THEMES[self.current_theme]["text"], 
                              wrap='word', bd=2, relief='solid')
        self.chat_box.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_box.config(state=tk.DISABLED)

    def setup_stats_tab(self):
        # Heading
        tk.Label(self.tab_stats, text="💫 Dashboard", 
               font=('Comic Sans MS', 18, 'bold'),
               bg=THEMES[self.current_theme]["bg"], 
               fg=THEMES[self.current_theme]["title"]).pack(pady=15)
        
        # Stats container with border
        stats_frame = tk.Frame(self.tab_stats, bg=THEMES[self.current_theme]["border"], 
                             bd=3, relief=tk.RIDGE)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        inner_stats = tk.Frame(stats_frame, bg=THEMES[self.current_theme]["highlight"])
        inner_stats.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Amigurumi stats
        tk.Label(inner_stats, text="🧸 Top Amigurumi", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"]).pack(anchor='w', padx=20, pady=(15, 5))
        
        self.amigurumi_stats = tk.Text(inner_stats, height=4, width=40, font=('Comic Sans MS', 12),
                                     bg=THEMES[self.current_theme]["chat_bg"], 
                                     fg=THEMES[self.current_theme]["text"], bd=0)
        self.amigurumi_stats.pack(fill=tk.X, padx=20, pady=5)
        
        # Color stats
        tk.Label(inner_stats, text="🎨 Color Usage", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"]).pack(anchor='w', padx=20, pady=(15, 5))
        
        self.color_stats = tk.Text(inner_stats, height=4, width=40, font=('Comic Sans MS', 12),
                                 bg=THEMES[self.current_theme]["chat_bg"], 
                                 fg=THEMES[self.current_theme]["text"], bd=0)
        self.color_stats.pack(fill=tk.X, padx=20, pady=5)
        
        # Emoji stats
        tk.Label(inner_stats, text="😊 Top Emojis", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"]).pack(anchor='w', padx=20, pady=(15, 5))
        
        self.emoji_stats = tk.Text(inner_stats, height=3, width=40, font=('Comic Sans MS', 12),
                                 bg=THEMES[self.current_theme]["chat_bg"], 
                                 fg=THEMES[self.current_theme]["text"], bd=0)
        self.emoji_stats.pack(fill=tk.X, padx=20, pady=5)
        
        # Session info
        tk.Label(inner_stats, text="📊 Overall Stats", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"]).pack(anchor='w', padx=20, pady=(15, 5))
        
        self.session_stats = tk.Text(inner_stats, height=3, width=40, font=('Comic Sans MS', 12),
                                   bg=THEMES[self.current_theme]["chat_bg"], 
                                   fg=THEMES[self.current_theme]["text"], bd=0)
        self.session_stats.pack(fill=tk.X, padx=20, pady=(5, 15))
        
        # Make all text widgets read-only
        for widget in (self.amigurumi_stats, self.color_stats, self.emoji_stats, self.session_stats):
            widget.config(state=tk.DISABLED)
        
        # Update button
        update_btn = tk.Button(self.tab_stats, text="✨ Update Stats", command=self.update_stats_display,
                             font=('Comic Sans MS', 12),
                             bg=THEMES[self.current_theme]["button"], 
                             fg=THEMES[self.current_theme]["button_text"],
                             padx=10, pady=5)
        update_btn.pack(pady=15)

    def setup_gallery_tab(self):
        # Heading
        tk.Label(self.tab_gallery, text="🧸 Amigurumi Gallery", 
               font=('Comic Sans MS', 18, 'bold'),
               bg=THEMES[self.current_theme]["bg"], 
               fg=THEMES[self.current_theme]["title"]).pack(pady=15)
        
        # Gallery container
        self.gallery_frame = tk.Frame(self.tab_gallery, bg=THEMES[self.current_theme]["bg"])
        self.gallery_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollable container
        gallery_canvas = tk.Canvas(self.gallery_frame, bg=THEMES[self.current_theme]["bg"], 
                                 highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=gallery_canvas.yview)
        self.scrollable_gallery = tk.Frame(gallery_canvas, bg=THEMES[self.current_theme]["bg"])
        
        self.scrollable_gallery.bind(
            "<Configure>",
            lambda e: gallery_canvas.configure(scrollregion=gallery_canvas.bbox("all"))
        )
        
        gallery_canvas.create_window((0, 0), window=self.scrollable_gallery, anchor="nw")
        gallery_canvas.configure(yscrollcommand=scrollbar.set)
        
        gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Refresh button
        refresh_btn = tk.Button(self.tab_gallery, text="🔄 Refresh Gallery", command=self.update_gallery,
                              font=('Comic Sans MS', 12),
                              bg=THEMES[self.current_theme]["button"], 
                              fg=THEMES[self.current_theme]["button_text"],
                              padx=10, pady=5)
        refresh_btn.pack(pady=10)

    def setup_settings_tab(self):
        # Heading
        tk.Label(self.tab_settings, text="🎀 Settings", 
               font=('Comic Sans MS', 18, 'bold'),
               bg=THEMES[self.current_theme]["bg"], 
               fg=THEMES[self.current_theme]["title"]).pack(pady=15)
        
        # Settings container
        settings_frame = tk.Frame(self.tab_settings, bg=THEMES[self.current_theme]["bg"])
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Theme selector
        theme_frame = tk.Frame(settings_frame, bg=THEMES[self.current_theme]["bg"])
        theme_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(theme_frame, text="🎨 Theme:", 
               font=('Comic Sans MS', 14),
               bg=THEMES[self.current_theme]["bg"], 
               fg=THEMES[self.current_theme]["fg"]).pack(side=tk.LEFT, padx=(0, 10))
        
        theme_names = list(THEMES.keys())
        self.theme_var = tk.StringVar(value=self.current_theme)
        theme_dropdown = ttk.Combobox(theme_frame, textvariable=self.theme_var, 
                                    values=theme_names, width=15, font=('Comic Sans MS', 12))
        theme_dropdown.pack(side=tk.LEFT)
        
        apply_theme_btn = tk.Button(theme_frame, text="Apply", command=self.change_theme,
                                  font=('Comic Sans MS', 12),
                                  bg=THEMES[self.current_theme]["button"], 
                                  fg=THEMES[self.current_theme]["button_text"])
        apply_theme_btn.pack(side=tk.LEFT, padx=10)
        
        # Stream settings
        stream_frame = tk.Frame(settings_frame, bg=THEMES[self.current_theme]["highlight"], 
                              bd=2, relief=tk.GROOVE, pady=10, padx=10)
        stream_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(stream_frame, text="📺 Stream Settings", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["title"]).pack(anchor='w', pady=(0, 10))
        
        self.live_var = tk.BooleanVar(value=True)
        live_check = tk.Checkbutton(stream_frame, text="Bot is currently live streaming", 
                                  variable=self.live_var, 
                                  font=('Comic Sans MS', 12),
                                  bg=THEMES[self.current_theme]["highlight"], 
                                  fg=THEMES[self.current_theme]["fg"],
                                  selectcolor=THEMES[self.current_theme]["highlight"],
                                  command=self.toggle_live_status)
        live_check.pack(anchor='w')
        
        # Bot settings
        bot_frame = tk.Frame(settings_frame, bg=THEMES[self.current_theme]["highlight"], 
                           bd=2, relief=tk.GROOVE, pady=10, padx=10)
        bot_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(bot_frame, text="🤖 Bot Settings", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["title"]).pack(anchor='w', pady=(0, 10))
        
        # Channel input
        channel_frame = tk.Frame(bot_frame, bg=THEMES[self.current_theme]["highlight"])
        channel_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(channel_frame, text="Twitch Channel:", 
               font=('Comic Sans MS', 12),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"]).pack(side=tk.LEFT, padx=(0, 10))
        
        self.channel_var = tk.StringVar(value=CHANNEL)
        channel_entry = tk.Entry(channel_frame, textvariable=self.channel_var, 
                               font=('Comic Sans MS', 12), width=15)
        channel_entry.pack(side=tk.LEFT)
        
        # Clear learned data
        clear_button = tk.Button(bot_frame, text="Clear Learned Emojis", 
                               font=('Comic Sans MS', 12),
                               bg=THEMES[self.current_theme]["button"], 
                               fg=THEMES[self.current_theme]["button_text"],
                               command=self.clear_emoji_data)
        clear_button.pack(anchor='w', pady=5)
        
        # About section
        about_frame = tk.Frame(settings_frame, bg=THEMES[self.current_theme]["highlight"], 
                             bd=2, relief=tk.GROOVE, pady=10, padx=10)
        about_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(about_frame, text="ℹ️ About", 
               font=('Comic Sans MS', 14, 'bold'),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["title"]).pack(anchor='w', pady=(0, 10))
        
        about_text = "Rena's Crochet Helper v1.0\n" + \
                    "Created with 💖 for amigurumi enthusiasts\n" + \
                    "© 2023 Kawaii Software"
        
        tk.Label(about_frame, text=about_text, 
               font=('Comic Sans MS', 12),
               bg=THEMES[self.current_theme]["highlight"], 
               fg=THEMES[self.current_theme]["fg"],
               justify=tk.LEFT).pack(anchor='w')

    def setup_label_dialog(self):
        self.label_dialog = tk.Toplevel(self.root)
        self.label_dialog.title("Name This Amigurumi")
        self.label_dialog.configure(bg=THEMES[self.current_theme]["bg"])
        self.center_window(self.label_dialog, 500, 400)
        self.label_dialog.withdraw()  # Hide initially
        
        # Preview
        self.preview_label = tk.Label(self.label_dialog, bg=THEMES[self.current_theme]["border"],
                                    bd=5, relief=tk.GROOVE)
        self.preview_label.pack(padx=20, pady=20)
        
        # Instructions
        tk.Label(self.label_dialog, text="What are you crocheting?", 
               font=('Comic Sans MS', 16, 'bold'),
               bg=THEMES[self.current_theme]["bg"], 
               fg=THEMES[self.current_theme]["fg"]).pack(pady=10)
        
        # Entry
        self.new_label_var = tk.StringVar()
        entry = tk.Entry(self.label_dialog, textvariable=self.new_label_var, 
                       font=('Comic Sans MS', 14), width=30)
        entry.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(self.label_dialog, bg=THEMES[self.current_theme]["bg"])
        btn_frame.pack(pady=20)
        
        save_btn = tk.Button(btn_frame, text="Save", command=self.save_new_label,
                           font=('Comic Sans MS', 12),
                           bg=THEMES[self.current_theme]["button"], 
                           fg=THEMES[self.current_theme]["button_text"],
                           padx=20, pady=5)
        save_btn.pack(side=tk.LEFT, padx=10)
        
        skip_btn = tk.Button(btn_frame, text="Skip", command=self.skip_labeling,
                           font=('Comic Sans MS', 12),
                           bg=THEMES[self.current_theme]["button"], 
                           fg=THEMES[self.current_theme]["button_text"],
                           padx=20, pady=5)
        skip_btn.pack(side=tk.LEFT, padx=10)

    def init_video_capture(self):
        # Try to connect to webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Webcam not available, using video file")
            cap = cv2.VideoCapture(FALLBACK_VIDEO_PATH)
            if not cap.isOpened():
                print("Video file not found, creating dummy capture")
                # Create a dummy black frame
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy, "No Camera Found", (160, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.latest_frame = dummy
                return None
        return cap

    def update_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # If it's a video file and reaches the end, loop back
                if not self.cap.isOpened() or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                self.latest_frame = frame.copy()
                
                # Resize for display
                frame = cv2.resize(frame, (640, 480))
                # Convert to PhotoImage
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update video label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        
        # Update floating emojis
        self.update_floating_emojis()
        
        # Schedule the next update
        self.root.after(33, self.update_frame)  # ~30 FPS

    def get_latest_frame(self):
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame.copy()

    def ask_user_to_label(self, frame):
        # Store the frame for learning
        self.stored_frame = frame.copy()
        
        # Prepare the preview image
        preview = cv2.resize(frame, (320, 240))
        img = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update preview and show dialog
        self.preview_label.imgtk = imgtk
        self.preview_label.configure(image=imgtk)
        self.new_label_var.set("")
        self.center_window(self.label_dialog, 400, 400)
        self.label_dialog.deiconify()

    def save_new_label(self):
        label = self.new_label_var.get().strip()
        if label and self.stored_frame is not None:
            learn_amigurumi(self.stored_frame, label)
            self.add_message_to_chat(f"Learned new amigurumi: {label} 📝")
            self.label_dialog.withdraw()
            self.stored_frame = None
            # Tell the bot that labeling is done
            if hasattr(bot, 'waiting_for_label'):
                bot.waiting_for_label = False
            # Update the gallery
            self.update_gallery()

    def skip_labeling(self):
        self.label_dialog.withdraw()
        self.stored_frame = None
        # Tell the bot that labeling is done
        if hasattr(bot, 'waiting_for_label'):
            bot.waiting_for_label = False

    def update_display(self, label, color_name, hex_color, friendly_name, progress, message):
        self.label_text.set(f"Detected: {label}")
        self.color_text.set(f"Yarn Color: {color_name}")
        self.friendly_color_text.set(f"({friendly_name})")
        self.progress_text.set(f"Progress: {progress}")
        
        # Update color swatch
        self.color_swatch.itemconfig(self.swatch_rect, fill=hex_color)
        
        # Add message to chat
        self.add_message_to_chat(message)
        
        # Add some random emojis to the emoji canvas
        self.add_floating_emoji()

    def add_message_to_chat(self, message):
        # Enable editing, add message, disable editing
        self.chat_box.config(state=tk.NORMAL)
        self.chat_box.insert(tk.END, f"{message}\n\n")
        self.chat_box.see(tk.END)
        self.chat_box.config(state=tk.DISABLED)

    def add_floating_emoji(self):
        if not self.canvas_emojis:
            return
            
        emojis = load_emojis()
        if not emojis:
            return
            
        # Add 1-3 random emojis
        count = random.randint(1, 3)
        for _ in range(count):
            emoji = random.choice(emojis)
            x = random.randint(10, 140)
            y = random.randint(5, 25)
            self.floating_emojis.append(FloatingEmoji(self.canvas_emojis, emoji, x, y))

    def update_floating_emojis(self):
        if not self.canvas_emojis:
            return
            
        # Update all floating emojis
        self.floating_emojis = [emoji for emoji in self.floating_emojis if emoji.update()]

    def update_stats_display(self):
        # Get stats data
        stats = load_stats()
        amigurumi_data = get_top_amigurumi()
        emoji_data = get_top_emojis()
        favorite_color = get_favorite_color()
        
        # Update amigurumi stats
        self.amigurumi_stats.config(state=tk.NORMAL)
        self.amigurumi_stats.delete(1.0, tk.END)
        if amigurumi_data:
            for i, (ami, count) in enumerate(amigurumi_data, 1):
                self.amigurumi_stats.insert(tk.END, f"{i}. {ami}: {count} times\n")
        else:
            self.amigurumi_stats.insert(tk.END, "No amigurumi data yet!")
        self.amigurumi_stats.config(state=tk.DISABLED)
        
        # Update color stats
        self.color_stats.config(state=tk.NORMAL)
        self.color_stats.delete(1.0, tk.END)
        self.color_stats.insert(tk.END, f"Most used color: {favorite_color}\n\n")
        if "colors_used" in stats and stats["colors_used"]:
            for color, count in sorted(stats["colors_used"].items(), key=lambda x: x[1], reverse=True)[:3]:
                self.color_stats.insert(tk.END, f"{color}: {count} times\n")
        self.color_stats.config(state=tk.DISABLED)
        
        # Update emoji stats
        self.emoji_stats.config(state=tk.NORMAL)
        self.emoji_stats.delete(1.0, tk.END)
        if emoji_data:
            for emoji, count in emoji_data:
                self.emoji_stats.insert(tk.END, f"{emoji}: {count} times  ")
        else:
            self.emoji_stats.insert(tk.END, "No emoji data yet!")
        self.emoji_stats.config(state=tk.DISABLED)
        
        # Update session stats
        self.session_stats.config(state=tk.NORMAL)
        self.session_stats.delete(1.0, tk.END)
        self.session_stats.insert(tk.END, f"Total sessions: {stats['sessions']}\n")
        self.session_stats.insert(tk.END, f"Total detections: {stats['detection_count']}\n")
        self.session_stats.insert(tk.END, f"Last session: {stats['last_session']}")
        self.session_stats.config(state=tk.DISABLED)

    def update_gallery(self):
        # Clear the gallery
        for widget in self.scrollable_gallery.winfo_children():
            widget.destroy()
        
        # Load amigurumi data
        data = load_amigurumi_data()
        if not data:
            label = tk.Label(self.scrollable_gallery, text="No amigurumi images yet!", 
                           font=('Comic Sans MS', 14),
                           bg=THEMES[self.current_theme]["bg"], 
                           fg=THEMES[self.current_theme]["fg"])
            label.pack(pady=20)
            return
        
        # Create frames for each type
        row_frame = None
        count = 0
        
        for label, image_paths in data.items():
            if not image_paths:
                continue
                
            # For every 2 items, create a new row
            if count % 2 == 0:
                row_frame = tk.Frame(self.scrollable_gallery, bg=THEMES[self.current_theme]["bg"])
                row_frame.pack(fill=tk.X, pady=10)
            
            # Create item frame
            item_frame = tk.Frame(row_frame, bg=THEMES[self.current_theme]["border"], 
                                 bd=3, relief=tk.RIDGE)
            item_frame.pack(side=tk.LEFT, padx=10)
            
            inner_frame = tk.Frame(item_frame, bg=THEMES[self.current_theme]["bg"])
            inner_frame.pack(padx=2, pady=2)
            
            # Title
            tk.Label(inner_frame, text=label, 
                   font=('Comic Sans MS', 12, 'bold'),
                   bg=THEMES[self.current_theme]["bg"], 
                   fg=THEMES[self.current_theme]["title"]).pack(pady=5)
            
            # Get the latest image
            latest_image = image_paths[-1]
            if os.path.exists(latest_image):
                img = cv2.imread(latest_image)
                img = cv2.resize(img, (200, 150))  # Thumbnail size
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                photo = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(inner_frame, image=photo, bg=THEMES[self.current_theme]["bg"])
                img_label.image = photo  # Keep a reference
                img_label.pack(padx=10, pady=5)
                
                # Show count
                count_data = load_amigurumi_count()
                count_text = f"Made {count_data.get(label, 0)} times"
                tk.Label(inner_frame, text=count_text, 
                       font=('Comic Sans MS', 10),
                       bg=THEMES[self.current_theme]["bg"], 
                       fg=THEMES[self.current_theme]["fg"]).pack(pady=5)
            
            count += 1

    def change_theme(self):
        new_theme = self.theme_var.get()
        if new_theme in THEMES:
            self.current_theme = new_theme
            self.apply_theme(new_theme)
            self.root.title(f"🧶 Rena's Crochet Monitor ({new_theme})")
            
            # Update background colors for all frames
            for frame in [self.tab_live, self.tab_stats, self.tab_gallery, self.tab_settings]:
                frame.configure(bg=THEMES[new_theme]["bg"])
            
            # Rebuild the UI for each tab
            self.setup_live_tab()
            self.setup_stats_tab()
            self.setup_gallery_tab()
            self.setup_settings_tab()
            
            # Also update the label dialog
            self.setup_label_dialog()

    def toggle_live_status(self):
        if hasattr(bot, 'is_live'):
            bot.is_live = self.live_var.get()
            status = "LIVE" if bot.is_live else "OFFLINE"
            self.add_message_to_chat(f"Bot is now {status} 🔄")

    def clear_emoji_data(self):
        if os.path.exists(EMOJI_FILE):
            os.remove(EMOJI_FILE)
        self.add_message_to_chat("Emoji database cleared! 🧹")

    def center_window(self, window, width, height):
        # Get screen dimensions
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set geometry
        window.geometry(f"{width}x{height}+{x}+{y}")

    def run(self):
        # Update video frame
        self.update_frame()
        
        # Update stats initially
        self.update_stats_display()
        
        # Update gallery initially
        self.update_gallery()
        
        # Start the main loop
        self.root.mainloop()
        
        # Cleanup
        if self.cap is not None:
            self.cap.release()

# === Main Program ===
if __name__ == "__main__":
    # Create app
    app = CrochetApp()
    
    # Create bot
    bot = Bot()
    
    # Run the bot in a separate thread
    import threading
    bot_thread = threading.Thread(target=asyncio.run, args=(bot.run_bot_logic(),))
    bot_thread.daemon = True
    bot_thread.start()
    
    # Connect to Twitch in another thread
    twitch_thread = threading.Thread(target=bot.run)
    twitch_thread.daemon = True
    twitch_thread.start()
    
    # Run the app
    app.run()
