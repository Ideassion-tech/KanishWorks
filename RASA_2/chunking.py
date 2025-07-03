import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the folder containing your PDFs
pdf_folder = "RASA_2\docs2"

# Recursive Character Text Splitter settings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

# Collect all chunks
all_chunks = []

# Loop through all PDF files in the folder
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Reading {filename}")
        reader = PdfReader(pdf_path)

        # Extract text from all pages
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Split the text into chunks
        chunks = text_splitter.split_text(full_text)
        
        # You may optionally store which file the chunks came from
        for chunk in chunks:
            all_chunks.append({
                "source_file": filename,
                "text": chunk
            })

print(f"Total Chunks: {len(all_chunks)}")
print("Sample chunk:", all_chunks[0])
