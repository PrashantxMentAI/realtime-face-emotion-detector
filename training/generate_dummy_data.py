import os
import csv
import random

def generate_mock_fer2013(output_path, num_samples=200):
    """Generates a mock CSV file formatted exactly like the real FER-2013 dataset.
    
    Format:
    emotion,pixels,Usage
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FER-2013 emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
    # Note: FERPlus maps these slightly differently, but this is the standard FER-2013 structure.
    emotions = [0, 1, 2, 3, 4, 5, 6]
    usages = ["Training"] * int(num_samples * 0.8) + ["PublicTest"] * int(num_samples * 0.1) + ["PrivateTest"] * int(num_samples * 0.1)
    # Ensure usage list matches sample count
    while len(usages) < num_samples:
        usages.append("Training")
    random.shuffle(usages)

    print(f"Generating mock dataset with {num_samples} samples...")
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["emotion", "pixels", "Usage"])
        
        for i in range(num_samples):
            emotion = random.choice(emotions)
            usage = usages[i]
            
            # Generate 2304 random pixel intensities (48x48 = 2304)
            pixels = [random.randint(0, 255) for _ in range(2304)]
            pixels_str = " ".join(map(str, pixels))
            
            writer.writerow([emotion, pixels_str, usage])
            
    print(f"Mock dataset successfully created at: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    generate_mock_fer2013(os.path.join(base_dir, "fer2013_dummy.csv"))
