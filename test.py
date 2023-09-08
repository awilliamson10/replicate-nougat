import replicate

MODEL_NAME = "awilliamson10/meta-nougat:3c18529626681ac442278dc89d739ea19ecb80a4b8f51e0ba9276e355c51c9f4"

def main():
    prediction = replicate.run(
        MODEL_NAME,
        input={"pdf_link": "https://s29.q4cdn.com/175625835/files/doc_downloads/test.pdf"},
    )
    print(prediction)

if __name__ == "__main__":
    main()