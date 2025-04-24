import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
import os
from length import get_fish_length, classification_curvature
from seg import segmentation_pipeline
import openpyxl
from openpyxl.drawing.image import Image as ExcelImage
import io

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        folder_label.config(text=f"Selected Folder: {folder_path}")
        process_button.config(state=tk.NORMAL)
        process_button.folder_path = folder_path

def on_drop(event):
    folder_path = event.data.strip()
    if os.path.isdir(folder_path):
        folder_label.config(text=f"Selected Folder: {folder_path}")
        process_button.config(state=tk.NORMAL)
        process_button.folder_path = folder_path
    else:
        messagebox.showerror("Error", "Dropped item is not a valid folder.")

def write_lengths_to_excel(folder_path, fish_lengths, filenames, curvatures):
    try:
        # Create a new workbook and select the active worksheet
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Fish Data"

        # Write the header row
        sheet.append(["Filename", "Fish Length (units)", "Curvature"])

        # Write the data rows
        for i, filename in enumerate(filenames):
            fish_length = fish_lengths[i] if i < len(fish_lengths) else "N/A"
            curvature = curvatures[i] if i < len(curvatures) else "N/A"
            sheet.append([filename, fish_length, curvature])

        # Calculate statistics
        if fish_lengths:
            median_length = sorted(fish_lengths)[len(fish_lengths) // 2]
            percentile_25 = sorted(fish_lengths)[int(len(fish_lengths) * 0.25)]
            percentile_75 = sorted(fish_lengths)[int(len(fish_lengths) * 0.75)]
        else:
            median_length = percentile_25 = percentile_75 = "N/A"

        if curvatures:
            median_curvature = sorted(curvatures)[len(curvatures) // 2]
            percentile_25_curvature = sorted(curvatures)[int(len(curvatures) * 0.25)]
            percentile_75_curvature = sorted(curvatures)[int(len(curvatures) * 0.75)]
            mean_curvature = sum(curvatures) / len(curvatures)
            std_dev_curvature = (sum((x - mean_curvature) ** 2 for x in curvatures) / len(curvatures)) ** 0.5
        else:
            median_curvature = percentile_25_curvature = percentile_75_curvature = mean_curvature = std_dev_curvature = "N/A"

        # Write the statistics
        sheet.append([])
        sheet.append(["Statistics"])
        sheet.append(["Median Length", median_length])
        sheet.append(["25th Percentile Length", percentile_25])
        sheet.append(["75th Percentile Length", percentile_75])
        sheet.append(["Median Curvature", median_curvature])
        sheet.append(["25th Percentile Curvature", percentile_25_curvature])
        sheet.append(["75th Percentile Curvature", percentile_75_curvature])
        sheet.append(["Mean Curvature", mean_curvature])
        sheet.append(["Standard Deviation Curvature", std_dev_curvature])

        # Add a boxplot to the Excel sheet
        import matplotlib.pyplot as plt

        # Create the boxplot for fish lengths
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.boxplot(fish_lengths, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.title("Fish Lengths")
        plt.ylabel("Length (units)")

        # Create the boxplot for curvatures
        plt.subplot(2, 2, 2)
        plt.boxplot(curvatures, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        plt.title("Curvatures")
        plt.ylabel("Curvature")

        # Plot the mean and standard deviation for fish lengths
        plt.subplot(2, 2, 3)
        if fish_lengths:
            mean_length = sum(fish_lengths) / len(fish_lengths)
            std_dev_length = (sum((x - mean_length) ** 2 for x in fish_lengths) / len(fish_lengths)) ** 0.5
            plt.boxplot(fish_lengths, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
            plt.axhline(mean_length, color='red', linestyle='--', label=f"Mean: {mean_length:.2f}", xmin=0.25, xmax=0.75)
            plt.axhline(mean_length + std_dev_length, color='green', linestyle='--', label=f"Std Dev: {std_dev_length:.2f}", xmin=0.25, xmax=0.75)
            plt.axhline(mean_length - std_dev_length, color='green', linestyle='--', xmin=0.25, xmax=0.75)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1)
        plt.title("Fish Lengths: Boxplot with Mean and Std Dev")
        plt.ylabel("Length (units)")

        # Plot the boxplot for curvatures with mean and standard deviation
        plt.subplot(2, 2, 4)
        if curvatures:
            mean_curvature = sum(curvatures) / len(curvatures)
            std_dev_curvature = (sum((x - mean_curvature) ** 2 for x in curvatures) / len(curvatures)) ** 0.5
            plt.boxplot(curvatures, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
            plt.axhline(mean_curvature, color='red', linestyle='--', label=f"Mean: {mean_curvature:.2f}")
            plt.axhline(mean_curvature + std_dev_curvature, color='green', linestyle='--', label=f"Std Dev: {std_dev_curvature:.2f}", xmin=0.25, xmax=0.75)
            plt.axhline(mean_curvature - std_dev_curvature, color='green', linestyle='--', xmin=0.25, xmax=0.75)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=1)

        plt.title("Curvatures: Boxplot with Mean and Std Dev")
        plt.ylabel("Curvature")

        # Save the plot to a BytesIO stream
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format='png', bbox_inches='tight')
        plt.close()
        img_stream.seek(0)

        # Add the image to the Excel sheet
        img = ExcelImage(img_stream)
        img.anchor = "E2"  # Position the image in the sheet
        sheet.add_image(img)

        # Save the workbook to the selected folder
        output_path = os.path.join(folder_path, "fish_data.xlsx")
        workbook.save(output_path)
        messagebox.showinfo("Excel Saved", f"Fish data saved to {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save Excel file: {e}")


def start_processing():
    folder_path = process_button.folder_path
    global stop_processing
    stop_processing = False

    # Get the selected processing options
    process_curvature = curvature_var.get()
    process_length = length_var.get()

    original_images, segmented_images, grown_images = segmentation_pipeline(folder_path)
    
    fish_lengths = []
    curvatures = []
    for i, segmented_image in enumerate(segmented_images):
        if stop_processing:
            break
        try:
            if process_length:
                result_image, fish_length = get_fish_length(segmented_image)
                fish_lengths.append(fish_length)
                curve_length_label.config(text=f"Fish Length: {fish_length:.2f} units")
                # Convert the result image to a format suitable for Tkinter
                result_image_tk = ImageTk.PhotoImage(Image.fromarray(result_image))
                image_label.config(image=result_image_tk)
                image_label.image = result_image_tk

            if process_curvature:
                masked_image, curvature = classification_curvature(original_images[i], grown_images[i])
                curvatures.append(curvature)
                curvature_label.config(text=f"Curvature: {curvature:.2f}")
                masked_image_tk = ImageTk.PhotoImage(Image.fromarray(masked_image))
                masked_image_label.config(image=masked_image_tk)
                masked_image_label.image = masked_image_tk

            root.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
    
    # Call the function after processing is complete
    filenames = [file_name for file_name in os.listdir(folder_path)]
    write_lengths_to_excel(folder_path, fish_lengths, filenames, curvatures)
    if not stop_processing:
        messagebox.showinfo("Done", "Processing is complete.")

def stop_processing_action():
    global stop_processing
    stop_processing = True
    messagebox.showinfo("Stopped", "Processing has been stopped.")

# Create the GUI
root = TkinterDnD.Tk()
root.title("Image Processing GUI")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack()

# Enable drag-and-drop functionality
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

folder_label = tk.Label(frame, text="No folder selected", wraplength=400)
folder_label.pack(pady=5)

select_button = tk.Button(frame, text="Select Folder", command=select_folder)
select_button.pack(pady=5)

# Add toggles for processing options
curvature_var = tk.BooleanVar(value=True)
length_var = tk.BooleanVar(value=True)

curvature_toggle = tk.Checkbutton(frame, text="Process Curvature", variable=curvature_var)
curvature_toggle.pack(pady=5)

length_toggle = tk.Checkbutton(frame, text="Process Length", variable=length_var)
length_toggle.pack(pady=5)

process_button = tk.Button(frame, text="Start Processing", state=tk.DISABLED, command=start_processing)
process_button.pack(pady=5)

stop_button = tk.Button(frame, text="Stop Processing", command=stop_processing_action)
stop_button.pack(pady=5)

image_label = tk.Label(frame)
image_label.pack(pady=10)

masked_image_label = tk.Label(frame)
masked_image_label.pack(pady=10)

curve_length_label = tk.Label(frame, text="")
curve_length_label.pack(pady=5)

curvature_label = tk.Label(frame, text="")
curvature_label.pack(pady=5)

root.mainloop()