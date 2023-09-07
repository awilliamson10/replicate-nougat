import replicate

MODEL_NAME = "awilliamson10/meta-nougat:42a683e961e164dc4647590a38a8571a11ee446d78b2f928c0abd3a21235d34a"

def main():
    prediction = replicate.run(
        MODEL_NAME,
        input={"pdf_link": "https://s29.q4cdn.com/175625835/files/doc_downloads/test.pdf"},
    )
    print(prediction)

if __name__ == "__main__":
    main()