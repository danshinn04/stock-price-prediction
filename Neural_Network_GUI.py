# NeuralNetworkGUI.py

import tkinter as tk
from tkinter import Canvas, Frame, Label, Entry, Button, messagebox

class Neural_Network_GUI:
    def __init__(self, master):
        self.master = master
        self.network_structure = []

        # Create the canvas to draw the neural network, this will expand to fill the frame
        self.network_canvas = Canvas(master, bg="white")
        self.network_canvas.pack(side="top", fill="both", expand=True)

        # Bind the configure event to dynamically adjust the canvas size
        self.master.bind("<Configure>", self.on_resize)
        
        # Add a frame for the network settings
        self.settings_frame = Frame(master)
        self.settings_frame.pack(side="bottom", fill="x")

        # Add entry to specify the number of neurons in the next layer
        Label(self.settings_frame, text="Neurons:").pack(side="left")
        self.neuron_entry = Entry(self.settings_frame)
        self.neuron_entry.pack(side="left")

        # Add a button to add the layer
        Button(self.settings_frame, text="Add Layer", command=self.add_layer).pack(side="left")

        # Draw the initial empty network
        self.draw_neural_network()

    def on_resize(self, event):
        # Redraw the neural network
        self.draw_neural_network()

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

    def draw_neural_network(self):
        self.network_canvas.delete("all")  # Clear the current network drawing

        # Get the current canvas dimensions
        canvas_width = self.network_canvas.winfo_width()
        canvas_height = self.network_canvas.winfo_height()

        # Determine the max width and height available for each neuron, including spacing
        max_layer_size = max(self.network_structure, default=1)
        max_width_per_neuron = canvas_width / (len(self.network_structure) + 1)
        max_height_per_neuron = canvas_height / (max_layer_size + 1)

        # Determine the size of each neuron, taking the smaller of width vs. height constraint
        neuron_diameter = min(max_width_per_neuron, max_height_per_neuron) / 2

        # Loop through each layer
        for layer_idx, layer_size in enumerate(self.network_structure):
            # Calculate the horizontal center of this layer
            layer_center_x = (layer_idx + 1) * (canvas_width / (len(self.network_structure) + 1))

            # Loop through each neuron in the layer
            for neuron_idx in range(layer_size):
                # Calculate vertical center position of each neuron
                neuron_center_y = (neuron_idx + 1) * (canvas_height / (layer_size + 1))

                # Draw the neuron as an oval
                self.network_canvas.create_oval(
                    layer_center_x - neuron_diameter / 2,
                    neuron_center_y - neuron_diameter / 2,
                    layer_center_x + neuron_diameter / 2,
                    neuron_center_y + neuron_diameter / 2,
                    fill="white", outline="black"
                )

                # Draw lines to next layer if not the last layer
                if layer_idx < len(self.network_structure) - 1:
                    next_layer_size = self.network_structure[layer_idx + 1]
                    next_layer_center_x = (layer_idx + 2) * (canvas_width / (len(self.network_structure) + 1))

                    # Draw a line from this neuron to each neuron in the next layer
                    for next_neuron_idx in range(next_layer_size):
                        next_neuron_center_y = (next_neuron_idx + 1) * (canvas_height / (next_layer_size + 1))

                        self.network_canvas.create_line(
                            layer_center_x, neuron_center_y,
                            next_layer_center_x, next_neuron_center_y
                        )

        # Update the canvas to reflect the new drawing
        self.network_canvas.update_idletasks()
