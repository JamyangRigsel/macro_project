"""Tkinter desktop application for macroinvertebrate image prediction."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import joblib
from PIL import Image, ImageTk

from src.config import MODEL_OUTPUT_DIR
from src.services.image_preprocessor import ImagePreprocessor


class MacroApp(tk.Tk):
    """Desktop GUI for macroinvertebrate image prediction."""

    def __init__(self, preprocessor: ImagePreprocessor, model_path: Path) -> None:
        super().__init__()

        self.title("Macroinvertebrate Image Analysis System")
        self.geometry("850x600")
        self.resizable(False, False)

        self.preprocessor = preprocessor
        self.model_path = model_path
        self.selected_file: str | None = None

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            self.model_ready = True
        else:
            self.model = None
            self.model_ready = False

        self.create_widgets()

    def create_widgets(self) -> None:
        """Create the main UI layout."""

        title_label = tk.Label(
            self,
            text="Macroinvertebrate Image Analysis System",
            font=("Arial", 20, "bold"),
        )
        title_label.pack(pady=15)

        subtitle_label = tk.Label(
            self,
            text="Select an image and predict its macroinvertebrate class",
            font=("Arial", 11),
        )
        subtitle_label.pack(pady=5)

        self.image_frame = tk.Frame(
            self,
            width=420,
            height=320,
            relief="solid",
            borderwidth=1,
        )
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)

        self.image_label = tk.Label(
            self.image_frame,
            text="No image selected",
            font=("Arial", 12),
        )
        self.image_label.pack(expand=True)

        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        choose_button = tk.Button(
            button_frame,
            text="Choose Image",
            command=self.choose_image,
            width=18,
            height=2,
        )
        choose_button.grid(row=0, column=0, padx=10)

        predict_button = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict_image,
            width=18,
            height=2,
        )
        predict_button.grid(row=0, column=1, padx=10)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_image,
            width=18,
            height=2,
        )
        clear_button.grid(row=0, column=2, padx=10)

        self.result_label = tk.Label(
            self,
            text="Prediction result will appear here.",
            font=("Arial", 14, "bold"),
        )
        self.result_label.pack(pady=20)

        if self.model_ready:
            status_text = "Model loaded successfully."
        else:
            status_text = "Demo mode: model not trained yet."

        self.status_label = tk.Label(
            self,
            text=status_text,
            font=("Arial", 10),
        )
        self.status_label.pack(pady=5)

    def choose_image(self) -> None:
        """Open file picker and display the selected image."""

        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        self.selected_file = file_path

        image = Image.open(file_path)
        image.thumbnail((390, 290))

        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

        self.result_label.configure(text="Image selected. Click Predict.")

    def predict_image(self) -> None:
        """Predict the selected image class."""

        if not self.selected_file:
            messagebox.showwarning(
                "No image selected",
                "Please choose an image first.",
            )
            return

        if not self.model_ready:
            self.result_label.configure(
                text="Demo prediction: Model not trained yet."
            )
            messagebox.showinfo(
                "Demo Mode",
                "The UI is working, but the trained model file was not found.\n\n"
                "Run `python -m src.main` first to train the model.",
            )
            return

        try:
            features = self.preprocessor.transform(self.selected_file).reshape(1, -1)
            prediction = self.model.predict(features)[0]

            if hasattr(self.model, "predict_proba"):
                confidence = self.model.predict_proba(features).max()
                result_text = (
                    f"Predicted class: {prediction} | "
                    f"Confidence: {confidence:.2%}"
                )
            else:
                result_text = f"Predicted class: {prediction}"

            self.result_label.configure(text=result_text)

        except ValueError as error:
            messagebox.showerror("Prediction Error", str(error))

    def clear_image(self) -> None:
        """Clear the selected image and reset the UI."""

        self.selected_file = None
        self.image_label.configure(image="", text="No image selected")
        self.image_label.image = None
        self.result_label.configure(text="Prediction result will appear here.")


def main() -> None:
    """Launch the Tkinter application."""

    model_path = MODEL_OUTPUT_DIR / "macro_classifier.joblib"
    preprocessor = ImagePreprocessor()

    app = MacroApp(preprocessor, model_path)
    app.mainloop()


if __name__ == "__main__":
    main()
