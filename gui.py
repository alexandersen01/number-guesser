import tkinter as tk
from PIL import ImageDraw, ImageTk
import PIL
import numpy as np
import torch
import numberguesser as ng

# Load model
model = ng.Numbers()
model.load_state_dict(torch.load('model.pth'))

label = None

# Dictionary to label all the classes
classes = {
    0: 'Zero',
    1: 'One',
    2: 'Two',
    3: 'Three',
    4: 'Four',
    5: 'Five',
    6: 'Six',
    7: 'Seven',
    8: 'Eight',
    9: 'Nine'
}

# Initialise GUI
window = tk.Tk()
window.title('Number Guesser')
window.geometry('600x600')

# Initialise canvas
canvas = tk.Canvas(window, width=500, height=500, bg='black')
canvas.pack()

# Initialise PIL image
img = PIL.Image.new('RGB', (500, 500), 'black')
draw = ImageDraw.Draw(img)

# Add clear button
def clear():
    canvas.delete('all')
    img.paste((0, 0, 0), [0, 0, 500, 500])
    #remove label
    global label
    if label:
        label.destroy()



# Function to classify image
def classify():
    # Convert image to MNIST format
    img_resized = img.resize((28, 28)).convert('L')
    img_np = np.array(img_resized)
    img_normalized = img_np / 255.0
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()

    # Get prediction
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs.data, 1)

    # Remove previous label if it exists
    global label
    if label:
        label.destroy()

    # Display prediction
    label = tk.Label(window, text=f'Prediction: {classes[predicted.item()]}', font=('Arial', 32))
    label.place(x=100, y=450)

# Function to draw on canvas
def paint(event):
    x = event.x
    y = event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='white')
    draw.rectangle((x - r, y - r, x + r, y + r), fill='white')

# Bind mouse events
canvas.bind('<B1-Motion>', paint)

# Add buttons
button_clear = tk.Button(window, text = 'Clear', font = ('Arial', 32), command = clear)
button_clear.place(x = 100, y = 500)

button_classify = tk.Button(window, text = 'Classify', font = ('Arial', 32), command = classify)
button_classify.place(x = 300, y = 500)

window.mainloop()
