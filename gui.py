import tkinter as tk
from PIL import Image, ImageTk
import threading
import cv2

class LiveGUI:
    def __init__(self):
        self.root = None
        self.thread = None
        self.image_label = None
        self.top_entry = None
        self.bottom_entry = None
        self._imgtk = None  # keep reference

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.root = tk.Tk()
        self.root.title("Live GUI")

        # Casella di testo sopra
        self.top_entry = tk.Entry(self.root, width=50)
        self.top_entry.pack(pady=5)

        # Immagine
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        # Casella di testo sotto
        self.bottom_entry = tk.Entry(self.root, width=50)
        self.bottom_entry.pack(pady=5)

        # Pulsanti
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        for text in ["Start", "Stop", "Reset"]:
            tk.Button(frame, text=text, command=lambda t=text: print(f"{t} clicked")).pack(side=tk.LEFT, padx=5)

        self.root.mainloop()

    def set_top_text(self, text):
        if self.top_entry:
            self.top_entry.delete(0, tk.END)
            self.top_entry.insert(0, text)

    def set_bottom_text(self, text):
        if self.bottom_entry:
            self.bottom_entry.delete(0, tk.END)
            self.bottom_entry.insert(0, text)

    def update_image(self, frame_bgr):
        if self.image_label and self.root:
            # Convert OpenCV BGR to RGB and then to ImageTk
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self._imgtk = ImageTk.PhotoImage(img)
            self.image_label.config(image=self._imgtk)
