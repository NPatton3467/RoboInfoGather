import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class PixelSelector:
    def __init__(self, root, img):
        self.root = root
        self.root.title("Image Pixel Selector")

        self.pixels = []
        self.image = None
        self.display_image = None
        self.photo = None

        # === Frame to contain scrollable canvas ===
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for image display
        self.canvas = tk.Canvas(self.canvas_frame, bg='black')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(fill=tk.X)

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # Frame inside canvas
        self.image_container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_container, anchor="nw")

        # Submit button
        self.submit_button = tk.Button(root, text="Submit", command=self.submit)
        self.submit_button.pack(pady=10)
        
        # Clear button
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)

        self.canvas.bind("<Configure>", self.on_canvas_configure)

        #self.load_image(img)
        self.root.after(0, lambda: self.load_image(img))

    def on_canvas_configure(self, event):
        # Update scroll region when canvas size changes
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_image(self,img):
        self.image = img
        self.display_image = Image.fromarray(self.image)
        self.photo = ImageTk.PhotoImage(self.display_image, master=self.root)
        
        # Image label
        self.image_label = tk.Label(self.image_container, image=self.photo)
        self.image_label.photo = self.photo
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.on_click)

        # Force update scroll region
        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def on_click(self, event):
        x, y = event.x, event.y
        if self.image is not None:
            height, width = self.image.shape[:2]
            if 0 <= x < width and 0 <= y < height:
                rgb = tuple(self.image[y, x])
                self.pixels.append((x, y))
                print(f"Clicked at ({x}, {y}) - RGB: {rgb}")

    def submit(self):
        print("\nSubmitted Pixel Values:")
        for (x, y) in self.pixels:
            print(f"({x}, {y})")
        self.root.quit()

    def clear(self):
        self.pixels = []
        print("\nCleared all Pixel Values")

    def shutdown(self):
        del(self.root)
        del(self.image)
        del(self.photo)
        del(self.canvas_frame)
        del(self.canvas)
        del(self.v_scroll)
        del(self.h_scroll)
        del(self.image_container)
        del(self.submit_button)
        del(self.clear_button)

if __name__ == "__main__":
    img = np.load('./debug/0/img_0.npy')
    root = tk.Tk()
    app = PixelSelector(root, img)
    root.mainloop()

    print("Selected Pixels: ", app.pixels)
