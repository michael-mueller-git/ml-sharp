
import gradio as gr
import subprocess
import os
import shutil
import time
import glob

def predict(image):
    if os.path.exists("/app/data/output"):
        shutil.rmtree("/app/data/output")
    # Ensure data directory exists
    os.makedirs("/app/data", exist_ok=True)
    
    input_path = "/app/data/input.jpg"
    
    # Save/Copy input image
    # image provided by gradio (type='filepath') is a temp path
    shutil.copy(image, input_path)

    # Run sharp command
    # sharp predict -i /app/data/input.jpg -o /app/data/output --render
    cmd = [
        "sharp", "predict",
        "-i", input_path,
        "-o", "/app/data/output",
        "--no-render"
    ]
    
    # Execute command
    try:
        t = time.time()
        print("Sharp started")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Sharp command took {round(time.time() - t, 3)} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Error running sharp: {e}")
        print(f"Stdout: {e.stdout.decode()}")
        print(f"Stderr: {e.stderr.decode()}")
        return None

    os.system("ls /app/data/output")
    data = "/app/data/output/input.ply"

    if os.path.exists(data):
        return data
        
    return None

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs=[gr.File(label="ply")],
    title="Sharp 3D View Synthesis",
    description="Upload an image to generate a 3D view ply."
)

if __name__ == "__main__":
    print("Sharp Monocular View Synthesis in Less Than a Second (https://github.com/apple/ml-sharp)")
    demo.launch(server_name="0.0.0.0", server_port=7860)
