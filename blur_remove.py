import requests

# URL of the FastAPI deblur endpoint
url = "https://83cf-14-139-34-151.ngrok-free.app/restore"  # Updated URL
  # Make sure there's only one slash

# Path to the input image
file_path = "SAMPLE.png"  # Replace with your actual image path

# Open the image file and send it as a POST request
with open(file_path, 'rb') as file:
    files = {'file': ('blurry_image.png', file, 'image/png')}  # (filename, file_object, MIME type)
    response = requests.post(url, files=files)

# Process the response
if response.status_code == 200:
    print("File processed successfully!")
    with open("deblurred_output.png", "wb") as output_file:
        output_file.write(response.content)
    print("Deblurred image saved as 'deblurred_output.png'.")
else:
    print("Failed to process the file.")
    print("Status Code:", response.status_code)
    print("Error:", response.text)
