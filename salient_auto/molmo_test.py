from molmo import MolmoModel

molmo = MolmoModel()

# Set test inputs
image_path = "teapot.jpg"
item="teapot"
text_prompt = "Point to the " + item

# Run model and print output
output_text = molmo.generate_response(image_path, text_prompt)
print("\n🔹 Molmo Output:", output_text)
coords = molmo.parse_coordinates(output_text)
if coords is not None:
    print(f"Parsed Coordinates: x = {coords[0]}, y = {coords[1]}")
else:
    print("No coordinates found in the output.")
