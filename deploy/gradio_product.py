import csv
import glob
import os
from pathlib import Path
import re
from uuid import uuid4
from clearml import Dataset, Model
import gradio as gr
import joblib
import time
from transformers import pipeline

class GradioApp():
    def __init__(self, transformer_model_id, sklearn_model_id):
        self.transformer_model_path = Path(Model(transformer_model_id).get_local_copy()).parent.absolute()
        self.sklearn_model_path = Model(sklearn_model_id).get_local_copy()

        self.sklearn_pipeline = joblib.load(self.sklearn_model_path)
        self.transformer_pipeline = pipeline("text-classification", model=self.transformer_model_path, device='cpu')

    def classify_transformer(self, sentence):
        start = time.time()
        sarcastic = self.transformer_pipeline(sentence)[0]
        time_taken = time.time() - start
        return f"LABEL: {sarcastic['label']}\nCERTAINTY: {sarcastic['score']:.2f}\nCOMPUTE TIME: {time_taken:.5f}"

    def classify_logistic(self, sentence):
        start = time.time()
        sarcastic = self.sklearn_pipeline.predict_proba([sentence])[0]
        time_taken = time.time() - start
        if sarcastic[0] > sarcastic[1]:
            label = "NORMAL"
            score = sarcastic[0]
        else:
            label = "SARCASTIC"
            score = sarcastic[1]
        return f"LABEL: {label}\nCERTAINTY: {score:.2f}\nCOMPUTE TIME: {time_taken:.5f}"

    def parse_output_to_label(self, output):
        if not output.startswith("LABEL"):
            return [output]
        # Extract the label and certainty value from the text
        result = re.search(r"LABEL: (\w+)\nCERTAINTY: ([\d\.]+)", output)

        # Check if the label is "NORMAL" or "SARCASTIC"
        if result:
            label = result.group(1)
            certainty = float(result.group(2))
            if label == "NORMAL":
                return [0]
            elif label == "SARCASTIC":
                return [1]

    def log_to_csv(self, text, model_output, csv_name, count, prefix=""):
        os.makedirs("flagged", exist_ok=True)
        log_filepath = Path("flagged") / (prefix + str(csv_name))
        csv_data = self.parse_output_to_label(model_output) + [text]

        with open(log_filepath, "a", newline="", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if os.stat(log_filepath).st_size > 0:
                pass
            else:
                writer.writerow(['label', 'comment'])
            writer.writerow(csv_data)

        new_count = count + 1
        return new_count, f"{new_count} labeled samples"

    def create_clearml_dataset_version(self, csv_filename, amount, counter):
        paths = glob.glob(str(Path("flagged") / f"*_{csv_filename}"))
        if paths:
            latest_clearml_dataset_id = Dataset.get(dataset_project="sarcasm_detector", dataset_name="kaggle_sarcasm").id
            print(f"{latest_clearml_dataset_id=}")
            updated_dataset = Dataset.create(
                dataset_project="sarcasm_detector",
                dataset_name="kaggle_sarcasm",
                parent_datasets=[latest_clearml_dataset_id]
            )
            [updated_dataset.add_files(path) for path in paths]
            updated_dataset.finalize(auto_upload=True)

            # remove the comment to clean up the labeled csv files
            #[os.remove(path) for path in paths]

            return f"{uuid4()}.csv", 0, "0 labeled samples"
        return csv_filename, amount, counter

    def demo(self):
        demo = gr.Blocks()
        with demo:
            # transformers_callback = ClearMLDatasetLogger()
            # logistic_callback = ClearMLDatasetLogger()
            amount_labeled_var = gr.State(0)
            csv_filename = gr.State(f"{uuid4()}.csv")

            with gr.Row():
                gr.HTML("<p style='text-align:center;'><img src='https://clear.ml/wp-content/uploads/2020/12/clearml-logo.svg' style='display:inline-block; margin:auto;' width='30%' /></p>")
            with gr.Row():
                with gr.Column():
                    text = gr.Textbox(label="Model input sentence")
                    b1 = gr.Button("Classify Sarcasm")
            with gr.Row():
                with gr.Column():
                    output_transformer = gr.Textbox(label="Model 1")
                    b2 = gr.Button("Model 1 was Wrong")
                with gr.Column():
                    output_logistic = gr.Textbox(label="Model 2")
                    b3 = gr.Button("Model 2 was Wrong")
            with gr.Row():
                with gr.Column():
                    counter = gr.Label("0 labeled samples")
                with gr.Column():
                    b4 = gr.Button(f"Package Labeled Samples")

            # This needs to be called at some point prior to the first call to callback.flag()
            # transformers_callback.setup([text, output_transformer, csv_filename, amount_labeled_var], "flagged_transformer")
            # logistic_callback.setup([text, output_logistic, csv_filename, amount_labeled_var], "flagged_logistic")

            # Run the models
            b1.click(self.classify_transformer, inputs=text, outputs=output_transformer)
            b1.click(self.classify_logistic, inputs=text, outputs=output_logistic)

            # Serve as a labeling tool
            b2.click(lambda *args: self.log_to_csv(*args, prefix="transformer_"), inputs=[text, output_transformer, csv_filename, amount_labeled_var], outputs=[amount_labeled_var, counter])
            b3.click(lambda *args: self.log_to_csv(*args, prefix="logistic_"), inputs=[text, output_logistic, csv_filename, amount_labeled_var], outputs=[amount_labeled_var, counter])

            # Package the current labels and ship them as a ClearML Dataset
            b4.click(self.create_clearml_dataset_version, inputs=[csv_filename, amount_labeled_var, counter], outputs=[csv_filename, amount_labeled_var, counter])

        demo.launch()