import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from PIL import ImageDraw, ImageFont
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from datetime import datetime  # Import datetime for timestamp
import random

# Load your pre-trained models
model1 = keras.models.load_model("arabic_model_adam5.keras")
model2 = keras.models.load_model("arabic_model_rmsprop1.keras")
model3 = keras.models.load_model("arabic_model_sgd.keras")

def contains_arabic(text):
    arabic_characters = "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"
    return any(char in arabic_characters for char in text)

# Modify the load_image function
def load_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            # Check if the file has a valid image extension
            allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
            if file_path.lower().endswith(allowed_extensions):
                image = Image.open(file_path)
                
                # Check if the image contains Arabic characters
                if contains_arabic(image.filename):
                    image.thumbnail((300, 300))
                    photo = ImageTk.PhotoImage(image)
                    image_label.config(image=photo)
                    image_label.image = photo

                    # Enable the Recognize Character button
                    recognize_button.config(state=tk.NORMAL)

                    # Save the file path for recognition
                    app.image_file_path = file_path
                else:
                    # Display popup error message
                    messagebox.showerror("Error", "Image does not contain Arabic characters. Please select an image with Arabic characters.")
            else:
                # Display popup error message
                messagebox.showerror("Error", "Unsupported file format. Please select a valid image file.")
    except FileNotFoundError:
        # Display popup error message
        messagebox.showerror("Error", "File not found.")
    except Exception as e:
        # Display popup error message
        messagebox.showerror("Error", f"Error loading image: {e}")


# Function to load and display an image
def recognize_character():
    try:
        # Clear the Listbox
        character_history_listbox.delete(0, tk.END)
        
        file_path = app.image_file_path
        if file_path:
            # Preprocess the image for model prediction
            img = load_img(file_path, target_size=(128, 128))
            img_array = img_to_array(img)
            img_array = img_array / 255.0  
            img_array = np.expand_dims(img_array, axis=0)

            # Use each pre-trained model for prediction
            prediction_adam = model1.predict(img_array)
            prediction_rmsprop = model2.predict(img_array)
            prediction_sgd = model3.predict(img_array)

            # Display results for the model trained with 'adam'
            confidence_adam = prediction_adam[0][np.argmax(prediction_adam)]
            recognized_character_adam, romanized_character_adam = display_result(prediction_adam, "Adam", confidence_adam)
            set_labels("Adam", recognized_character_adam, romanized_character_adam, confidence_adam, alphabet_label_adam, accuracy_label_adam)
            update_character_history("Adam", recognized_character_adam, romanized_character_adam, confidence_adam)

            # Display results for the model trained with 'rmsprop'
            confidence_rmsprop = prediction_rmsprop[0][np.argmax(prediction_rmsprop)]
            recognized_character_rmsprop, romanized_character_rmsprop = display_result(prediction_rmsprop, "RMSprop", confidence_rmsprop)
            set_labels("RMSprop", recognized_character_rmsprop, romanized_character_rmsprop, confidence_rmsprop, alphabet_label_rmsprop, accuracy_label_rmsprop)
            update_character_history("RMSprop", recognized_character_rmsprop, romanized_character_rmsprop, confidence_rmsprop)

            # Display results for the model trained with 'sgd'
            confidence_sgd = prediction_sgd[0][np.argmax(prediction_sgd)]
            recognized_character_sgd, romanized_character_sgd = display_result(prediction_sgd, "SGD", confidence_sgd)
            set_labels("SGD", recognized_character_sgd, romanized_character_sgd, confidence_sgd, alphabet_label_sgd, accuracy_label_sgd)
            update_character_history("SGD", recognized_character_sgd, romanized_character_sgd, confidence_sgd)

    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"Error recognizing character: {e}")

# Function to set labels based on recognition results
def set_labels(optimizer_name, recognized_character, romanized_character,
               confidence, alphabet_label, accuracy_label):
    # Choose the appropriate labels for the current model
    if optimizer_name == "Adam":
        alphabet_label.config(text=f"Recognized Character (Adam): {recognized_character} ({romanized_character})")
        accuracy_label.config(text=f"Confidence (Adam): {confidence * 100:.2f}%")
    elif optimizer_name == "RMSprop":
        alphabet_label.config(text=f"Recognized Character (RMSprop): {recognized_character} ({romanized_character})")
        accuracy_label.config(text=f"Confidence (RMSprop): {confidence * 100:.2f}%")
    elif optimizer_name == "SGD":
        alphabet_label.config(text=f"Recognized Character (SGD): {recognized_character} ({romanized_character})")

# Function to display results for a specific model
def display_result(prediction, optimizer_name, confidence):
    recognized_character = get_predicted_character(prediction)
    romanized_character = get_romanized_character(recognized_character)

    # Print suitable text based on recognition confidence
    if confidence >= 0.7:
        print("High confidence in character recognition.")
    elif confidence >= 0.5:
        print("Moderate confidence in character recognition.")
    else:
        print("Low confidence in character recognition.")

    # Choose the appropriate labels for the current model
    if optimizer_name == "Adam":
        alphabet_label_adam.config(text=f"Recognized Character (Adam): {recognized_character} ({romanized_character})")
        accuracy_label_adam.config(text=f"Confidence (Adam): {confidence * 100:.2f}%")
    elif optimizer_name == "RMSprop":
        alphabet_label_rmsprop.config(text=f"Recognized Character (RMSprop): {recognized_character} ({romanized_character})")
        accuracy_label_rmsprop.config(text=f"Confidence (RMSprop): {confidence * 100:.2f}%")
    elif optimizer_name == "SGD":
        alphabet_label_sgd.config(text=f"Recognized Character (SGD): {recognized_character} ({romanized_character})")
        accuracy_label_sgd.config(text=f"Confidence (SGD): {confidence * 100:.2f}%")

    return recognized_character, romanized_character

# Print suitable text based on recognition confidence
    if confidence >= 0.7:
        print("High confidence in character recognition.")
    elif confidence >= 0.5:
        print("Moderate confidence in character recognition.")
    else:
        print("Low confidence in character recognition.")

# Function to get the predicted character based on the model's output
def get_predicted_character(prediction):

    alphabet = "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"
    predicted_index = np.argmax(prediction)
    return alphabet[predicted_index]

# Function to get the Romanized version of the recognized character
def get_romanized_character(character):
    romanized_mapping = {
        "أ": "Alif",
        "ب": "Ba",
        "ت": "Ta",
        "ث": "Tsa",
        "ج": "Jim",
        "ح": "Ha",
        "خ": "Kho",
        "د": "Dal",
        "ذ": "Dzal",
        "ر": "Ro",
        "ز": "Zai",
        "س": "Sin",
        "ش": "Syin",
        "ص": "Sod",
        "ض": "Dhod",
        "ط": "Tha",
        "ظ": "Zha",
        "ع": "Ain",
        "غ": "Ghain",
        "ف": "Fa",
        "ق": "Qof",
        "ك": "Kaf",
        "ل": "Lam",
        "م": "Meem",
        "ن": "Noon",
        "ه": "Ha",
        "و": "Waw",
        "ي": "Ya"
    }
    return romanized_mapping.get(character, "Unknown")

# Function to clear all history with confirmation
def clear_all_history():
    confirmation = messagebox.askquestion("Clear All History", "Do you really want to delete all the history?", icon='warning')
    if confirmation == 'yes':
        character_history.clear()
        character_history_listbox.delete(0, tk.END)

# Create a list to store recognized characters and timestamps
character_history = []
    
# Function to update character history
def update_character_history(optimizer_name, recognized_character, romanized_character, accuracy):
    try:
        #file_name = app.image_file_path.split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{optimizer_name} - {recognized_character} ({romanized_character}) - Accuracy: {accuracy:.2%} - {timestamp}" "\n"
        character_history.append(entry)

        # Limit the history to the last 10 entries (adjust as needed)
        if len(character_history) > 10:
            character_history.pop(0)

        # Update the Listbox widget
        character_history_listbox.delete(0, tk.END)
        for item in character_history:
            character_history_listbox.insert(tk.END, item)
            character_history_listbox.insert(tk.END, '\n')
    except Exception as e:
        print(f"Error updating character history: {e}")
        
# Toggle full-screen mode
def toggle_fullscreen():
    state = app.attributes('-fullscreen')
    app.attributes('-fullscreen', not state)

# Exit full-screen mode
def exit_fullscreen():
    app.attributes('-fullscreen', False)

# Change the mouse cursor when it enters the button
def on_enter(event):
    load_button.config(cursor="hand2", bg="#e09f3e", fg="white")

# Change the mouse cursor when it leaves the button
def on_leave(event):
    load_button.config(cursor="", bg="maroon", fg="white")

# Change the mouse cursor when it enters the button
def on_enter_recognize(event):
    recognize_button.config(cursor="hand2", bg="#e09f3e", fg="white")

# Change the mouse cursor when it leaves the button
def on_leave_recognize(event):
    recognize_button.config(cursor="", bg="maroon", fg="white")

# Change the mouse cursor when it enters the clear button
def on_clear_enter(event): 
    clear_button.config(cursor="hand2", bg="silver", fg="black")

# Change the mouse cursor when it leaves the clear button
def on_clear_leave(event): 
    clear_button.config(cursor="", bg="dark gray", fg="black")

# Change the mouse cursor when it enters the clear button
def on_clear_history_enter(event): 
    clear_all_button.config(cursor="hand2", bg="silver", fg="black")

# Change the mouse cursor when it leaves the clear button
def on_clear_history_leave(event): 
    clear_all_button.config(cursor="", bg="dark gray", fg="black")
    
# Change the mouse cursor and color for the exit button
def on_enter_minimize(event):
    minimize_button.config(cursor="hand2", bg="sea green")

# Change the mouse cursor when it leaves the exit button
def on_leave_minimize(event):
    minimize_button.config(cursor="", bg="maroon")

# Change the mouse cursor and color for the exit button
def on_exit_button_enter(event):
    exit_button.config(cursor="hand2", bg="sea green")

# Change the mouse cursor when it leaves the exit button
def on_exit_button_leave(event):
    exit_button.config(cursor="", bg="red")

# Change the mouse cursor when it enters the button
def on_enter_trivia(event):
    trivia_button.config(cursor="hand2", bg="#e09f3e", fg="white")

# Change the mouse cursor when it leaves the button
def on_leave_trivia(event):
    trivia_button.config(cursor="", bg="maroon", fg="white")

# Function to clear the displayed image and information
def clear_display():
    confirmation = messagebox.askquestion("Clear All History", "Do you really want to delete all the history?", icon='warning')
    if confirmation == 'yes':
        # Clear the displayed image
        image_label.config(image=None)

        # Clear the recognized character and Romanized version labels for all optimizers
        alphabet_label_adam.config(text="Recognized Character (Adam):")
        accuracy_label_adam.config(text="Confidence (Adam):")

        alphabet_label_rmsprop.config(text="Recognized Character (RMSprop):")
        accuracy_label_rmsprop.config(text="Confidence (RMSprop):")

        alphabet_label_sgd.config(text="Recognized Character (SGD):")
        accuracy_label_sgd.config(text="Confidence (SGD):")

        # Disable the Recognize Character button after clearing the display
        recognize_button.config(state=tk.DISABLED)

        # Clear the file path for recognition
        app.image_file_path = None

        # Explicitly set the image attribute to None to release the reference
        image_label.image = None

# Minimize the application window
def minimize_window():
    app.iconify()

# Function to exit the application
def exit_application():
    app.destroy()

# Minimize or restore the application window
def toggle_minimize():
    current_state = app.attributes('-fullscreen')
    if current_state:
        app.attributes('-fullscreen', False)
        app.geometry('600x600')  
    else:
        app.attributes('-fullscreen', True)
        app.geometry('')  
    app.update()  

# Create the main application window
app = tk.Tk()
app.title("Arabic Character Recognizer")

# Set the window to full screen
app.attributes("-fullscreen", True)

# Create a frame for the background image
bg_frame = tk.Frame(app, bg="maroon")
bg_frame.pack(fill=tk.BOTH, expand=True, pady=0)  

# Load background image
bg_image = Image.open("ss.png")  
bg_image = bg_image.resize((app.winfo_screenwidth(), app.winfo_screenheight()))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(app, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Create a frame above all three frames
frame_above_all = tk.Frame(app, bg="maroon")
frame_above_all.pack(side=tk.TOP, fill=tk.BOTH) 

# Create a title label for the main frame
main_title_label = tk.Label(app, text="ARABIC CHARACTER RECOGNITION",
                            bg="maroon", fg="white", font=("Castellar", 20, "bold"))
main_title_label.place(relx=0.5, rely=0.01, anchor=tk.N)

# Create frames to organize labels for each model
frame_upload = tk.Frame(app, bg="maroon")
frame_recognition = tk.Frame(app, bg="maroon")
frame_history = tk.Frame(app, bg="maroon")

# Pack the frames
frame_upload.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
frame_upload.place(relx=0.5, rely=0.09, anchor=tk.N)
frame_recognition.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
frame_recognition.place(relx=0.25, rely=0.60, anchor=tk.N)
frame_history.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
frame_history.place(relx=0.75, rely=0.60, anchor=tk.N)

# Dictionary mapping Arabic characters to their Romanized equivalents
character_mapping = {
    "أ": "Alif",
    "ب": "Ba",
    "ت": "Ta",
    "ث": "Tsa",
    "ج": "Jim",
    "ح": "Ha",
    "خ": "Kho",
    "د": "Dal",
    "ذ": "Dzal",
    "ر": "Ro",
    "ز": "Zai",
    "س": "Sin",
    "ش": "Syin",
    "ص": "Sod",
    "ض": "Dhod",
    "ط": "Tha",
    "ظ": "Zha",
    "ع": "Ain",
    "غ": "Ghain",
    "ف": "Fa",
    "ق": "Qof",
    "ك": "Kaf",
    "ل": "Lam",
    "م": "Meem",
    "ن": "Noon",
    "ه": "Ha",
    "و": "Waw",
    "ي": "Ya"
}

def trivia_question():
    # Randomly select a character for the trivia question
    random_character = random.choice(list(character_mapping.keys()))
    
    # Retrieve the Romanized equivalent from the dictionary
    correct_answer = character_mapping[random_character]
    
    # Generate the trivia question
    question = f"What is the Romanized equivalent for the Arabic character '{random_character}'?"
    
    # Display the trivia question in a popup
    user_answer = simpledialog.askstring("Arabic Trivia", question)

    # Check the user's answer
    if user_answer and user_answer.lower() == correct_answer.lower():
        messagebox.showinfo("Correct!", "Congratulations! You answered correctly.")
    else:
        messagebox.showinfo("Incorrect", f"Sorry, the correct answer is '{correct_answer}'.")

# Frame 1: Upload Image and Recognize
frame_upload_label = tk.Label(frame_upload, text="Upload Image",
                              font=("Castellar", 16, "bold"), fg="white", bg="maroon")
frame_upload_label.pack(pady=10)

load_button_icon_path = "upload.png"  
load_button_icon = Image.open(load_button_icon_path)
load_button_icon = load_button_icon.resize((30, 30), Image.LANCZOS) 
load_button_icon = ImageTk.PhotoImage(load_button_icon)

load_button = tk.Button(frame_upload, text="Upload Image", command=load_image, bg="maroon", fg="white", font=("Castellar", 16, "bold"), image=load_button_icon, compound="left")
load_button.image = load_button_icon
load_button.pack(pady=10)

# Create a label to display the image using an icon
image_icon_path = "image.png"  
image_icon = Image.open(image_icon_path)
image_icon = image_icon.resize((128, 128), Image.LANCZOS)  
image_icon = ImageTk.PhotoImage(image_icon)

image_label = tk.Label(frame_upload, image=image_icon, bg="gainsboro")
image_label.image = image_icon
image_label.pack(side=tk.TOP)

recognize_button = tk.Button(frame_upload, text="Recognize Character", command=recognize_character, state=tk.DISABLED, bg="maroon", fg="white", font=("Castellar", 16, "bold"))
recognize_button.pack(pady=10, anchor='center')

# Create a button to clear the display
clear_button = tk.Button(frame_upload, text="Clear", command=clear_display, bg="gainsboro", fg="#4f0205", font=("Castellar", 16, "bold"))
clear_button.pack(pady=10, anchor='center')

# Frame 2: Recognized Characters, Romanized, and Accuracy
frame_recognition_label = tk.Label(frame_recognition, text="Recognition Results", font=("Castellar", 16, "bold"), fg="white", bg="maroon")
frame_recognition_label.pack(pady=10)

# Labels for Adam
alphabet_label_adam = tk.Label(frame_recognition, text="Recognized Character (Adam):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
alphabet_label_adam.pack()

accuracy_label_adam = tk.Label(frame_recognition, text="Confidence (Adam):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
accuracy_label_adam.pack()

# Labels for RMSprop
alphabet_label_rmsprop = tk.Label(frame_recognition, text="Recognized Character (RMSprop):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
alphabet_label_rmsprop.pack()

accuracy_label_rmsprop = tk.Label(frame_recognition, text="Confidence (RMSprop):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
accuracy_label_rmsprop.pack()

# Labels for SGD
alphabet_label_sgd = tk.Label(frame_recognition, text="Recognized Character (SGD):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
alphabet_label_sgd.pack()

accuracy_label_sgd = tk.Label(frame_recognition, text="Confidence (SGD):", font=("Castellar", 14, "bold"), fg="#a46e51", bg="maroon")
accuracy_label_sgd.pack()

# Function to set labels based on recognition results
def set_labels(optimizer_name, recognized_character, romanized_character, confidence, alphabet_label, accuracy_label):
    # Choose the appropriate labels for the current model
    if optimizer_name == "Adam":
        alphabet_label.config(text=f"Recognized Character (Adam): {recognized_character} ({romanized_character})")
        confidence_info = get_confidence_info(confidence)
        accuracy_label.config(text=f"Confidence (Adam): {confidence * 100:.2f}% {confidence_info}")
    elif optimizer_name == "RMSprop":
        alphabet_label.config(text=f"Recognized Character (RMSprop): {recognized_character} ({romanized_character})")
        confidence_info = get_confidence_info(confidence)
        accuracy_label.config(text=f"Confidence (RMSprop): {confidence * 100:.2f}% {confidence_info}")
    elif optimizer_name == "SGD":
        alphabet_label.config(text=f"Recognized Character (SGD): {recognized_character} ({romanized_character})")
        confidence_info = get_confidence_info(confidence)
        accuracy_label.config(text=f"Confidence (SGD): {confidence * 100:.2f}% {confidence_info}")

# Function to get confidence information (high, low, moderate)
def get_confidence_info(confidence):
    if confidence >= 0.7:
        return "(High Confidence)"
    elif confidence >= 0.5:
        return "(Moderate Confidence)"
    else:
        return "(Low Confidence)"

# Frame 3: History
frame_history_label = tk.Label(frame_history, text="Character History", font=("Castellar", 16, "bold"), fg="white", bg="maroon")
frame_history_label.pack(pady=10)

# Create a Listbox widget for character history
character_history_listbox = tk.Listbox(frame_history, height=5, width=50, font=("Castellar", 12), bg="#e09f3e", selectbackground="#3F826D")
character_history_listbox.pack(pady=10)

clear_all_button = tk.Button(frame_history, text="Clear All History", command=clear_all_history, relief=tk.GROOVE, bd=3, font=("Castellar", 16, "bold"), bg="gainsboro", fg="#4f0205")
clear_all_button.pack(pady=10, anchor='center')

# Add an exit button
exit_button = tk.Button(app, text="X", command=exit_fullscreen, bg="red", fg="white", font=("Castellar", 12, "bold"))
exit_button.place(x=app.winfo_screenwidth() - 30, y=0, width=50, height=30)

# Create frames to organize labels for each model
frame_adam = tk.Frame(app, bg="maroon")
frame_rmsprop = tk.Frame(app, bg="maroon")
frame_sgd = tk.Frame(app, bg="maroon")

# Pack the frames
frame_adam.pack(pady=10)
frame_rmsprop.pack(pady=10)
frame_sgd.pack(pady=10)

# Bind events to change the mouse cursor
load_button.bind("<Enter>", on_enter)
load_button.bind("<Leave>", on_leave)

# Bind events to change the mouse cursor
recognize_button.bind("<Enter>", on_enter_recognize)
recognize_button.bind("<Leave>", on_leave_recognize)

# Bind events to change the mouse cursor for the clear button
clear_button.bind("<Enter>", on_clear_enter)
clear_button.bind("<Leave>", on_clear_leave)

# Bind events to change the mouse cursor for the clear button
clear_all_button.bind("<Enter>", on_clear_history_enter)
clear_all_button.bind("<Leave>", on_clear_history_leave)

# Update the trivia button command to call the modified trivia_question function
trivia_button = tk.Button(app, text="Arabic Trivia", command=trivia_question, bg="maroon", fg="white", font=("Castellar", 12, "bold"))
trivia_button.place(x=0, y=0, width=150, height=30)  # Use place method for precise placement

# Bind events to change the mouse cursor for the trivia button
trivia_button.bind("<Enter>", on_enter_trivia)
trivia_button.bind("<Leave>", on_leave_trivia)

# Add a minimize button
minimize_button = tk.Button(app, text="_", command=toggle_minimize, bg="maroon", fg="white", font=("Arial", 12, "bold"))
minimize_button.place(x=app.winfo_screenwidth() - 60, y=0, width=30, height=30)

# Bind events to change the mouse cursor for the trivia button
minimize_button.bind("<Enter>", on_enter_minimize)
minimize_button.bind("<Leave>", on_leave_minimize)

# Add an exit button
exit_button = tk.Button(app, text="X", command=exit_application, bg="red", fg="white", font=("Arial", 12, "bold"))
exit_button.place(x=app.winfo_screenwidth() - 30, y=0, width=30, height=30)

# Bind events to change the mouse cursor and color for the exit button
exit_button.bind("<Enter>", on_exit_button_enter)
exit_button.bind("<Leave>", on_exit_button_leave)

app.mainloop()
