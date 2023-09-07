import replicate

MODEL_NAME = "awilliamson10/meta-nougat:a9786e4af4a416a20bc10165dae536492b7e72ea2698c7dd4923c89c2737209f"

def main():
    prediction = replicate.run(
        MODEL_NAME,
        input={"pdf_link": "https://s29.q4cdn.com/175625835/files/doc_downloads/test.pdf"},
    )
    print(prediction)

if __name__ == "__main__":
    main()