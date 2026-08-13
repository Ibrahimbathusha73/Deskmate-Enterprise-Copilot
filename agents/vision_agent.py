def vision_agent(image_path: str, question: str) -> dict:
    return {
        "answer": "Vision agent is not yet implemented. Please set up the HF Inference token and model endpoints.",
        "status": "not_implemented"
    }

if __name__ == "__main__":
    print(vision_agent("fake_path.jpg", "what is this?"))
