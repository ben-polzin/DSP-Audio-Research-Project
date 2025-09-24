import tkinter as tk
from tkinter import filedialog
from scipy.io import wavfile


# Function to handle the "Select Sweep Response" button click event
def select_sweep_response():
    # Open a file dialog to select a .wav file
    file_path = filedialog.askopenfilename(
        title="Select a .wav file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
    )

    # Check if a file was selected
    if file_path:
        # Update the label with the selected file name
        result_label.config(text=f"Selected Sweep Response: {file_path}", fg="green")

        # Read the .wav file using the selected file path
        samplerateRS, response = wavfile.read(file_path)
        print(f"Samplerate: {samplerateRS}")
        print(f"Response: {response}")
    else:
        # Update the label with an error message
        result_label.config(text="No file selected for Sweep Response.", fg="red")


# Function to handle the "Select Sweep" button click event
def select_sweep():
    # Open a file dialog to select a .wav file
    file_path = filedialog.askopenfilename(
        title="Select a .wav file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
    )

    # Check if a file was selected
    if file_path:
        # Update the label with the selected file name
        result_label.config(text=f"Selected Sweep: {file_path}", fg="green")

        # Read the .wav file using the selected file path
        samplerateRS, response = wavfile.read(file_path)
        print(f"Samplerate: {samplerateRS}")
        print(f"Response: {response}")
    else:
        # Update the label with an error message
        result_label.config(text="No file selected for Sweep.", fg="red")


# Create the main application window
root = tk.Tk()
root.title("Sweep Selector")  # Set the window title

# Create a frame to hold the buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=200, padx=200)

# Create the "Select Sweep Response" button
select_response_button = tk.Button(
    button_frame,
    text="Select Sweep Response",
    command=select_sweep_response
)
select_response_button.pack(side=tk.LEFT, padx=50)  # Place it on the left with padding

# Create the "Select Sweep" button
select_sweep_button = tk.Button(
    button_frame,
    text="Select Sweep",
    command=select_sweep
)
select_sweep_button.pack(side=tk.LEFT, padx=50)  # Place it next to the first button

# Create a label to display the result or error message
result_label = tk.Label(root, text="", fg="black")  # Default text color is black
result_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()