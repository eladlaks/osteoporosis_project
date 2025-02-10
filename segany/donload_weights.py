import requests

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
output_path = "segany/sam_vit_h_4b8939.pth"

response = requests.get(url, stream=True)
with open(output_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print("Download complete.")
