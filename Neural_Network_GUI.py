import tkinter as tk
from tkinter import Canvas, Frame, Label, Entry, Button, messagebox, Toplevel
from tkinter import Scrollbar
class Neural_Network_GUI:
    def __init__(self, master):
        self.master = master
        self.network_structure = []

        # Create the canvas to draw the neural network, this will expand to fill the frame
        self.network_canvas = Canvas(master, bg="white", height=200)
        self.network_canvas.pack(side="top", fill="both", expand=True)

        self.v_scroll = Scrollbar(self.master, orient='vertical', command=self.network_canvas.yview)
        self.network_canvas.configure(yscrollcommand=self.v_scroll.set)

        # Pack the scrollbar to the right of the canvas
        self.v_scroll.pack(side='right', fill='y')
        self.network_canvas.pack(side='left', fill='both', expand=True)

        # Add a frame for the network settings
        self.settings_frame = Frame(master)
        self.settings_frame.pack(side="bottom", fill="x")

        # Add entry to specify the number of neurons in the next layer
        Label(self.settings_frame, text="Neurons:").pack(side="left")
        self.neuron_entry = Entry(self.settings_frame)
        self.neuron_entry.pack(side="left")

        # Add a button to add the layer
        Button(self.settings_frame, text="Add Layer", command=self.add_layer).pack(side="left")

        # Button to finalize the network structure
        Button(self.settings_frame, text="Finalize Network", command=self.finalize_network).pack(side="left")
        self.network_canvas.bind("<Configure>", self.on_resize)
        # Draw the initial empty network
        self.draw_neural_network()


    def on_resize(self, event):
        self.network_canvas.configure(scrollregion=self.network_canvas.bbox("all"))


    def add_layer(self):
        try:
            neurons = int(self.neuron_entry.get())
            if neurons <= 0:
                messagebox.showerror("Error", "Number of neurons must be positive")
                return

            self.network_structure.append(neurons)
            self.draw_neural_network()
        except ValueError:
            messagebox.showerror("Error", "Invalid number of neurons")

    def calculate_neuron_size_and_spacing(self, canvas_width, canvas_height, network_structure):
        # Calculate the maximum number of neurons in a layer
        max_neurons = max(network_structure) if network_structure else 0

        # Calculate the size of a neuron based on the canvas height and the number of neurons
        neuron_size = min(30, canvas_height // (max_neurons + 1))  # Set a max size for the neurons

        # Calculate the spacing based on the canvas width and the number of layers
        horizontal_spacing = canvas_width // (len(network_structure) + 1)
        vertical_spacing = canvas_height // (max_neurons + 1)

        return neuron_size, horizontal_spacing, vertical_spacing
    def add_scrollbars(self):
    # Add vertical and horizontal scrollbars to the canvas
        self.v_scroll = tk.Scrollbar(self.master, orient="vertical", command=self.network_canvas.yview)
        self.h_scroll = tk.Scrollbar(self.master, orient="horizontal", command=self.network_canvas.xview)
        self.network_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.h_scroll.pack(side="bottom", fill="x")
        self.network_canvas.pack(side="left", fill="both", expand=True)

    def draw_neural_network(self):
        self.network_canvas.delete("all")  # Clear the current network drawing
        
        # Define margins and spacing
        top_margin = 20
        bottom_margin = 20
        left_margin = 50
        right_margin = 50
        vertical_spacing = 60
        horizontal_spacing = 80
        
        # Determine the canvas width and height
        canvas_width = self.network_canvas.winfo_width()
        canvas_height = self.network_canvas.winfo_height()
        
        # Determine the number of layers and maximum number of neurons in any layer
        num_layers = len(self.network_structure)
        max_neurons = max(self.network_structure, default=1)
        
        # Calculate the neuron size
        neuron_radius = min(20, ((canvas_height - top_margin - bottom_margin) / max_neurons) / 2)
        
        # Calculate spacing between neurons and layers
        layer_spacing = (canvas_width - left_margin - right_margin) / (num_layers + 1)
        neuron_spacing = (canvas_height - top_margin - bottom_margin) / (max_neurons + 1)
        
        # Loop through each layer
        for layer_index, layer_size in enumerate(self.network_structure):
            x = left_margin + (layer_index + 1) * layer_spacing
            
            # Draw each neuron in the layer
            for neuron_index in range(layer_size):
                y = top_margin + (neuron_index + 1) * neuron_spacing
                self.network_canvas.create_oval(x - neuron_radius, y - neuron_radius,
                                                x + neuron_radius, y + neuron_radius,
                                                fill="white", outline="black")
            
            # Draw connections from the current layer to the next layer
            if layer_index < num_layers - 1:
                next_layer_size = self.network_structure[layer_index + 1]
                for neuron_index in range(layer_size):
                    y1 = top_margin + (neuron_index + 1) * neuron_spacing
                    for next_neuron_index in range(next_layer_size):
                        y2 = top_margin + (next_neuron_index + 1) * neuron_spacing
                        self.network_canvas.create_line(x + neuron_radius, y1,
                                                        x + layer_spacing - neuron_radius, y2)
        
        # Set the canvas to scroll if elements are drawn outside of the visible area
        self.network_canvas.config(scrollregion=self.network_canvas.bbox("all"))


    def finalize_network(self):
        # Maybe open a new window or update a part of the GUI to confirm the network structure
        confirmation_window = Toplevel(self.master)
        confirmation_window.title("Confirm Network Structure")
        Label(confirmation_window, text="Your network structure has been set.").pack()
        Button(confirmation_window, text="OK", command=confirmation_window.destroy).pack()

    def get_network_structure(self):
        return self.network_structure
