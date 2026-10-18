import argparse
from lib.describe_image import image_description

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="Path to the image file")
    parser.add_argument("--query", type=str, help="Query text to describe the image")

    args = parser.parse_args()

    if not args.image or not args.query:
        print("Error: Both --image and --query arguments are required.")
        return

    # Call the describe_image function from the lib module
    response = image_description(args.image, args.query)
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")

if __name__ == "__main__":
    main()
