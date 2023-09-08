import re
import uuid
from functools import partial

import fitz
import requests
import torch
from cog import BasePredictor, Input
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
from tqdm import tqdm


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Loading model...")
        self.model = NougatModel.from_pretrained("./model/base/").to(torch.bfloat16)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    def get_pdf(self, pdf_link):
        unique_filename = f"input/downloaded_paper_{uuid.uuid4().hex}.pdf"
        response = requests.get(pdf_link)

        if response.status_code == 200:
            with open(unique_filename, "wb") as pdf_file:
                pdf_file.write(response.content)
        else:
            print("Failed to download the PDF.")
        return unique_filename

    def predict(
        self,
        pdf_link: str = Input(description="Link to the PDF to be annotated"),
    ) -> str:
        pdf_path = self.get_pdf(pdf_link)
        try:
            dataset = LazyDataset(
                pdf_path,
                partial(self.model.encoder.prepare_input, random_padding=False),
            )
        except fitz.fitz.FileDataError:
            return {"predictions": "Failed to load the PDF."}
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        predictions = []
        page_num = 0
        for idx, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = self.model.inference(image_tensors=sample)
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    print(
                        "Processing file %s with %i pages"
                        % (dataset.name, dataset.size)
                    )
                page_num += 1
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                    continue
                if model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        print(f"Skipping page {page_num} due to repetitions.")
                        predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{idx+j+1}]\n\n"
                        )
                else:
                    predictions.append(output)
            if is_last_page:
                formatted_output = "".join(predictions).strip()
                formatted_output = re.sub(r"\n{3,}", "\n\n", formatted_output).strip()

        return formatted_output


        




        
